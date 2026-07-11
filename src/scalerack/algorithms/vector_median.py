import numpy as np

from scalerack.algorithms.box import build_coverage_weights
from scalerack.algorithms.registry import register
from scalerack.common.footprint import gather_footprint
from scalerack.common.resample import run_pipeline
from scalerack.image_io import ImageInput

# Block size
MAX_DISTANCE_ELEMENTS = 4 * 1024 * 1024
DEDUPLICATION_THRESHOLD = 256


def weighted_vector_median(samples: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Return the observed vector with the lowest weighted distance sum.

    Repeated vectors are collapsed and scores are evaluated in bounded blocks.
    The worst-case work remains quadratic, as an exact vector median requires,
    but peak temporary memory is linear.
    """

    # Merge repeated colors by adding their coverage.
    if len(samples) >= DEDUPLICATION_THRESHOLD:
        unique, first_indices, inverse = np.unique(
            samples, axis=0, return_index=True, return_inverse=True
        )
        if len(unique) < len(samples):
            unique_coverage = np.bincount(inverse, weights=coverage)
            first_occurrence_order = np.argsort(first_indices)
            samples = unique[first_occurrence_order]
            coverage = unique_coverage[first_occurrence_order]

    # Terms shared by every block of candidate vectors.
    sample_count = len(samples)
    block_size = max(1, MAX_DISTANCE_ELEMENTS // sample_count)
    squared_norms = np.einsum("ij,ij->i", samples, samples)
    best_index = 0
    best_score = np.inf

    # Score each candidate by its coverage-weighted distance to all samples.
    for start in range(0, sample_count, block_size):
        stop = min(start + block_size, sample_count)

        # ||a - b||² = ||a||² + ||b||² - 2a·b
        distances = squared_norms[start:stop, None] + squared_norms[None, :]
        distances -= 2.0 * samples[start:stop] @ samples.T
        np.maximum(distances, 0.0, out=distances)
        np.sqrt(distances, out=distances)

        scores = distances @ coverage
        block_index = int(np.argmin(scores))
        if scores[block_index] < best_score:
            best_index = start + block_index
            best_score = scores[block_index]

    return samples[best_index]


@register()
def vector_median(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Downscale by selecting the vector median of each source footprint.

    The vector median is the source pixel whose color vector has the smallest
    coverage-weighted sum of Euclidean distances to the other pixels in an
    output pixel's exact source footprint. Unlike channel-wise medians, it
    always returns a color that occurred in the source image.
    """

    def compute(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
        input_height, input_width = values.shape[:2]
        if output_width > input_width or output_height > input_height:
            raise ValueError("vector_median only supports downscaling.")

        row_indices, row_weights = build_coverage_weights(input_height, output_height)
        col_indices, col_weights = build_coverage_weights(input_width, output_width)
        result = np.empty((output_height, output_width, values.shape[2]), dtype=values.dtype)

        for output_y, (source_rows, row_coverage) in enumerate(
            zip(row_indices, row_weights, strict=True)
        ):
            for output_x, (source_columns, col_coverage) in enumerate(
                zip(col_indices, col_weights, strict=True)
            ):
                samples, coverage = gather_footprint(
                    values, source_rows, row_coverage, source_columns, col_coverage
                )

                result[output_y, output_x] = weighted_vector_median(samples, coverage)

        return result

    return run_pipeline(image, compute, factor=factor, width=width, height=height)
