from typing import cast

import numpy as np

from scalerack.image_io import ImageT, from_array, to_array
from scalerack.validation import resolve_output_size


def nearest(
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageT:
    """Scale by copying the nearest source pixel.

    Preserves exact pixel values; the right choice for masks, label maps,
    and deliberately blocky display of low-resolution art.
    """
    array = to_array(image)
    output_height, output_width = resolve_output_size(
        array.shape[0], array.shape[1], factor, width, height
    )
    row_indices = compute_nearest_indices(array.shape[0], output_height)
    col_indices = compute_nearest_indices(array.shape[1], output_width)
    result = array[row_indices][:, col_indices]
    return cast(ImageT, from_array(result, image))


def compute_nearest_indices(input_size: int, output_size: int) -> np.ndarray:
    """Map each output pixel center to the closest source pixel index."""
    centers = (np.arange(output_size) + 0.5) * (input_size / output_size)
    return np.minimum(centers.astype(np.int64), input_size - 1)
