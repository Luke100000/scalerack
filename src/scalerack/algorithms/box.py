import numpy as np

from scalerack.algorithms.registry import register
from scalerack.common.resample import filter_axis, normalize_rows, run_pipeline
from scalerack.image_io import ImageInput


@register()
def box(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Scale by averaging each output pixel's exact source footprint.

    The best-behaved choice for downscaling (area interpolation); upscaling
    degenerates to nearest-neighbor blocks.
    """

    def compute(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
        input_height, input_width = values.shape[:2]
        row_indices, row_weights = build_coverage_weights(input_height, output_height)
        col_indices, col_weights = build_coverage_weights(input_width, output_width)
        filtered = filter_axis(values, row_indices, row_weights, axis=0)
        return filter_axis(filtered, col_indices, col_weights, axis=1)

    return run_pipeline(image, compute, factor=factor, width=width, height=height)


def build_coverage_weights(input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-coverage indices and weights: each output pixel averages its exact footprint."""
    scale = output_size / input_size
    left_edges = np.arange(output_size) / scale
    right_edges = (np.arange(output_size) + 1) / scale
    first = np.floor(left_edges).astype(np.int64)
    footprint = 1.0 / scale
    edges_align = input_size % output_size == 0 or output_size % input_size == 0
    tap_count = int(np.ceil(footprint)) + (not edges_align)
    positions = first[:, None] + np.arange(tap_count)[None, :]
    coverage = np.minimum(right_edges[:, None], positions + 1) - np.maximum(
        left_edges[:, None], positions
    )
    weights = np.clip(coverage, 0.0, 1.0)
    indices = np.clip(positions, 0, input_size - 1)
    return indices, normalize_rows(weights)
