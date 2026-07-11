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
def eagle2x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with the classic Eagle corner-rounding rules."""
    return run_expansion(image, expand_eagle2x)


@register(factor=3)
def eagle3x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 3x with the community Eagle3x extension of the Eagle rules."""
    return run_expansion(image, expand_eagle3x)


def expand_eagle2x(values: np.ndarray) -> np.ndarray:
    """Apply the Eagle rules once."""
    height, width, channels = values.shape
    center = values
    corners = eagle_corner_rules(values)

    result = np.empty((height * 2, width * 2, channels), dtype=values.dtype)
    for (row, column), (condition, replacement) in zip(
        ((0, 0), (0, 1), (1, 0), (1, 1)), corners, strict=True
    ):
        result[row::2, column::2] = choose_pixels(condition, replacement, center)
    return result


def expand_eagle3x(values: np.ndarray) -> np.ndarray:
    """Apply the Eagle3x rules once: Eagle corners, plus edge midpoints where both
    adjacent corner rules fire (they share a neighbor, so their colors agree)."""
    height, width, channels = values.shape
    center = values
    corners = eagle_corner_rules(values)
    top_left, top_right, bottom_left, bottom_right = (condition for condition, _ in corners)
    up_left, up_right, down_left, down_right = (replacement for _, replacement in corners)

    result = np.empty((height * 3, width * 3, channels), dtype=values.dtype)
    result[0::3, 0::3] = choose_pixels(top_left, up_left, center)
    result[0::3, 1::3] = choose_pixels(top_left & top_right, up_left, center)
    result[0::3, 2::3] = choose_pixels(top_right, up_right, center)
    result[1::3, 0::3] = choose_pixels(top_left & bottom_left, up_left, center)
    result[1::3, 1::3] = center
    result[1::3, 2::3] = choose_pixels(top_right & bottom_right, up_right, center)
    result[2::3, 0::3] = choose_pixels(bottom_left, down_left, center)
    result[2::3, 1::3] = choose_pixels(bottom_left & bottom_right, down_left, center)
    result[2::3, 2::3] = choose_pixels(bottom_right, down_right, center)
    return result


def eagle_corner_rules(
    values: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return (condition, replacement) per corner: a corner takes the neighbor
    color when its three adjacent neighbors are mutually equal."""
    up, down, left, right = extract_edge_neighbors(values)
    up_left, up_right, down_left, down_right = extract_corner_neighbors(values)

    top_left = pixels_equal(left, up_left) & pixels_equal(up_left, up)
    top_right = pixels_equal(up, up_right) & pixels_equal(up_right, right)
    bottom_left = pixels_equal(left, down_left) & pixels_equal(down_left, down)
    bottom_right = pixels_equal(right, down_right) & pixels_equal(down_right, down)

    return (
        (top_left, up_left),
        (top_right, up_right),
        (bottom_left, down_left),
        (bottom_right, down_right),
    )
