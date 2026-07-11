import numpy as np

from scalerack.algorithms.registry import register
from scalerack.common.neighborhoods import (
    choose_pixels,
    extract_corner_neighbors,
    extract_edge_neighbors,
    pixels_equal,
    run_expansion,
)
from scalerack.image_io import ImageInput


@register(factor=2)
def scale2x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with the Scale2x (EPX) neighborhood rules."""
    return run_expansion(image, expand_scale2x)


@register(factor=3)
def scale3x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 3x with the Scale3x neighborhood rules."""
    return run_expansion(image, expand_scale3x)


@register(factor=4)
def scale4x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 4x by applying Scale2x twice."""
    return run_expansion(image, expand_scale4x)


def expand_scale2x(values: np.ndarray) -> np.ndarray:
    """Apply the Scale2x rules once."""
    height, width, channels = values.shape
    center = values
    up, down, left, right = extract_edge_neighbors(values)

    left_eq_up = pixels_equal(left, up)
    up_eq_right = pixels_equal(up, right)
    left_eq_down = pixels_equal(left, down)
    down_eq_right = pixels_equal(down, right)

    top_left = left_eq_up & ~up_eq_right & ~left_eq_down
    top_right = up_eq_right & ~left_eq_up & ~down_eq_right
    bottom_left = left_eq_down & ~left_eq_up & ~down_eq_right
    bottom_right = down_eq_right & ~left_eq_down & ~up_eq_right

    result = np.empty((height * 2, width * 2, channels), dtype=values.dtype)
    result[0::2, 0::2] = choose_pixels(top_left, left, center)
    result[0::2, 1::2] = choose_pixels(top_right, right, center)
    result[1::2, 0::2] = choose_pixels(bottom_left, left, center)
    result[1::2, 1::2] = choose_pixels(bottom_right, right, center)
    return result


def expand_scale3x(values: np.ndarray) -> np.ndarray:
    """Apply the Scale3x rules once."""
    height, width, channels = values.shape
    center = values
    up, down, left, right = extract_edge_neighbors(values)
    up_left, up_right, down_left, down_right = extract_corner_neighbors(values)

    left_eq_up = pixels_equal(left, up)
    up_eq_right = pixels_equal(up, right)
    left_eq_down = pixels_equal(left, down)
    down_eq_right = pixels_equal(down, right)

    corner_tl = left_eq_up & ~up_eq_right & ~left_eq_down
    corner_tr = up_eq_right & ~left_eq_up & ~down_eq_right
    corner_bl = left_eq_down & ~left_eq_up & ~down_eq_right
    corner_br = down_eq_right & ~left_eq_down & ~up_eq_right

    edge_top = (corner_tl & ~pixels_equal(center, up_right)) | (
        corner_tr & ~pixels_equal(center, up_left)
    )
    edge_left = (corner_tl & ~pixels_equal(center, down_left)) | (
        corner_bl & ~pixels_equal(center, up_left)
    )
    edge_right = (corner_tr & ~pixels_equal(center, down_right)) | (
        corner_br & ~pixels_equal(center, up_right)
    )
    edge_bottom = (corner_bl & ~pixels_equal(center, down_right)) | (
        corner_br & ~pixels_equal(center, down_left)
    )

    result = np.empty((height * 3, width * 3, channels), dtype=values.dtype)
    result[0::3, 0::3] = choose_pixels(corner_tl, left, center)
    result[0::3, 1::3] = choose_pixels(edge_top, up, center)
    result[0::3, 2::3] = choose_pixels(corner_tr, right, center)
    result[1::3, 0::3] = choose_pixels(edge_left, left, center)
    result[1::3, 1::3] = center
    result[1::3, 2::3] = choose_pixels(edge_right, right, center)
    result[2::3, 0::3] = choose_pixels(corner_bl, left, center)
    result[2::3, 1::3] = choose_pixels(edge_bottom, down, center)
    result[2::3, 2::3] = choose_pixels(corner_br, right, center)
    return result


def expand_scale4x(values: np.ndarray) -> np.ndarray:
    """Scale4x is defined as Scale2x applied twice."""
    return expand_scale2x(expand_scale2x(values))
