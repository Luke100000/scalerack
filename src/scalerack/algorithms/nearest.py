import numpy as np

from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput, as_image_input


@register()
def nearest(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Scale by copying the nearest source pixel.

    Preserves exact pixel values; the right choice for masks, label maps,
    and deliberately blocky display of low-resolution art.
    """
    image_input = as_image_input(image)
    array = image_input.numpy()
    output_width, output_height = image_input.get_target_dimensions(width, height, factor)
    row_indices = compute_nearest_indices(array.shape[0], output_height)
    col_indices = compute_nearest_indices(array.shape[1], output_width)
    result = array[row_indices][:, col_indices]
    return image_input.from_numpy(result)


def compute_nearest_indices(input_size: int, output_size: int) -> np.ndarray:
    """Map each output pixel center to the closest source pixel index."""
    centers = (np.arange(output_size) + 0.5) * (input_size / output_size)
    return np.minimum(centers.astype(np.int64), input_size - 1)
