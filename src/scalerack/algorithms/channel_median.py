import numpy as np

from scalerack.algorithms.box import build_coverage_weights
from scalerack.algorithms.registry import register
from scalerack.common.footprint import gather_footprint
from scalerack.common.resample import run_pipeline
from scalerack.image_io import ImageInput


def weighted_channel_median(samples: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    # Sort every channel independently and carry each sample's coverage with it.
    order = np.argsort(samples, axis=0)
    sorted_samples = np.take_along_axis(samples, order, axis=0)
    sorted_coverage = coverage[order]
    cumulative_coverage = np.cumsum(sorted_coverage, axis=0)

    # Select the first value that reaches half of the footprint's coverage.
    channels = np.arange(samples.shape[1])
    upper = np.argmax(cumulative_coverage >= 0.5, axis=0)
    median = sorted_samples[upper, channels]

    # At an exact halfway split, use the midpoint of the two central values.
    exact_half = np.isclose(cumulative_coverage[upper, channels], 0.5)
    has_successor = upper + 1 < len(samples)
    midpoint = exact_half & has_successor
    successors = sorted_samples[np.minimum(upper + 1, len(samples) - 1), channels]
    median[midpoint] = (median[midpoint] + successors[midpoint]) / 2
    return median


@register()
def channel_median(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Downscale with a coverage-weighted median computed per channel.

    Each output channel is independently reduced to the median of its exact
    source footprint. This robustly rejects isolated channel outliers, but can
    construct a color vector that did not occur in the source image.
    """

    def compute(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
        input_height, input_width = values.shape[:2]
        if output_width > input_width or output_height > input_height:
            raise ValueError("channel_median only supports downscaling.")

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
                result[output_y, output_x] = weighted_channel_median(samples, coverage)

        return result

    return run_pipeline(image, compute, factor=factor, width=width, height=height)
