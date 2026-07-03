from collections.abc import Callable
from typing import cast

import numpy as np

from scalerack.image_io import ImageT, from_array, restore_dtype, to_array
from scalerack.kernels import Kernel
from scalerack.validation import resolve_output_size

ComputeFunction = Callable[[np.ndarray, int, int], np.ndarray]


def resample_with_kernel(
    image: ImageT,
    kernel: Kernel,
    *,
    factor: float | None,
    width: int | None,
    height: int | None,
) -> ImageT:
    """Validate the request, filter both axes with the kernel, restore the representation."""

    def compute(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
        input_height, input_width = values.shape[:2]
        row_indices, row_weights = build_kernel_weights(input_height, output_height, kernel)
        col_indices, col_weights = build_kernel_weights(input_width, output_width, kernel)
        filtered = filter_axis(values, row_indices, row_weights, axis=0)
        return filter_axis(filtered, col_indices, col_weights, axis=1)

    return run_pipeline(image, compute, factor=factor, width=width, height=height)


def run_pipeline(
    image: ImageT,
    compute: ComputeFunction,
    *,
    factor: float | None,
    width: int | None,
    height: int | None,
) -> ImageT:
    """Common flow: validate, convert to float, compute, cast back, restore representation."""
    array = to_array(image)
    output_height, output_width = resolve_output_size(
        array.shape[0], array.shape[1], factor, width, height
    )
    if (output_height, output_width) == array.shape[:2]:
        return cast(ImageT, from_array(array.copy(), image))
    # float32 carries uint8 content losslessly and halves the filtering cost;
    # float inputs keep their own precision.
    compute_dtype = np.float32 if array.dtype == np.uint8 else array.dtype
    values = ensure_channel_axis(array).astype(compute_dtype)
    filtered = compute(values, output_height, output_width)
    result = drop_channel_axis(restore_dtype(filtered, array.dtype), array.ndim)
    return cast(ImageT, from_array(result, image))


def build_kernel_weights(
    input_size: int, output_size: int, kernel: Kernel
) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-pixel source indices and normalized kernel weights for one axis.

    For downscaling the kernel is widened by the inverse scale so it acts as a
    low-pass filter (standard kernel-scaling); edges are handled by clamping
    indices, which replicates border pixels.
    """
    scale = output_size / input_size
    filter_scale = max(1.0, 1.0 / scale)
    support = kernel.support * filter_scale
    centers = (np.arange(output_size) + 0.5) / scale - 0.5
    first = np.ceil(centers - support).astype(np.int64)
    tap_count = int(np.floor(2 * support)) + 2
    positions = first[:, None] + np.arange(tap_count)[None, :]
    weights = kernel.evaluate((positions - centers[:, None]) / filter_scale)
    indices = np.clip(positions, 0, input_size - 1)
    return indices, normalize_rows(weights)


def normalize_rows(weights: np.ndarray) -> np.ndarray:
    """Normalize each row of weights to sum to one (guarding degenerate zero rows)."""
    totals = weights.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals == 0, 1.0, totals)
    return weights / safe_totals


def filter_axis(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray, *, axis: int
) -> np.ndarray:
    """Apply per-output-pixel weighted gathering along one axis."""
    moved = np.moveaxis(values, axis, 0)
    output_size, tap_count = indices.shape
    typed_weights = weights.astype(values.dtype, copy=False)
    result = np.zeros((output_size, *moved.shape[1:]), dtype=values.dtype)
    for tap in range(tap_count):
        tap_weights = typed_weights[:, tap].reshape(-1, *([1] * (moved.ndim - 1)))
        result += tap_weights * moved[indices[:, tap]]
    return np.moveaxis(result, 0, axis)


def ensure_channel_axis(array: np.ndarray) -> np.ndarray:
    """View grayscale (H, W) input as (H, W, 1) so all compute paths are 3-D."""
    if array.ndim == 2:
        return array[:, :, None]
    return array


def drop_channel_axis(array: np.ndarray, original_ndim: int) -> np.ndarray:
    """Undo :func:`ensure_channel_axis` for grayscale inputs."""
    if original_ndim == 2:
        return array[:, :, 0]
    return array
