from collections.abc import Callable
from typing import cast

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.image_io import ImageT, from_array, to_array
from scalerack.resample import drop_channel_axis, ensure_channel_axis

ExpandFunction = Callable[[np.ndarray], np.ndarray]


@register
def scale2x(image: ImageT) -> ImageT:
    """Enlarge pixel art exactly 2x with the Scale2x (EPX) neighborhood rules.

    Smooths diagonals by copying matching neighbors, never by blending;
    made for sprites and low-color art, not photographs.
    """
    return run_epx(image, expand_scale2x)


@register
def scale3x(image: ImageT) -> ImageT:
    """Enlarge pixel art exactly 3x with the Scale3x neighborhood rules."""
    return run_epx(image, expand_scale3x)


@register
def scale4x(image: ImageT) -> ImageT:
    """Enlarge pixel art exactly 4x by applying Scale2x twice."""
    return run_epx(image, expand_scale4x)


def run_epx(image: ImageT, expand: ExpandFunction) -> ImageT:
    """Expand with the given rule set and restore the input representation."""
    array = to_array(image)
    values = ensure_channel_axis(array)
    result = drop_channel_axis(expand(values), array.ndim)
    return cast(ImageT, from_array(result, image))


def expand_scale2x(values: np.ndarray) -> np.ndarray:
    """Apply the Scale2x rules once, mapping (H, W, C) to (2H, 2W, C).

    Clean-room implementation from the published rule set (scale2x.it/algorithm).
    """
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
    """Apply the Scale3x rules once, mapping (H, W, C) to (3H, 3W, C)."""
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


def extract_edge_neighbors(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the (up, down, left, right) neighbors with replicated borders."""
    padded = pad_edges(values)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return up, down, left, right


def extract_corner_neighbors(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the diagonal neighbors with replicated borders."""
    padded = pad_edges(values)
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]
    return up_left, up_right, down_left, down_right


def pad_edges(values: np.ndarray) -> np.ndarray:
    """Replicate the outermost pixels by one in each spatial direction."""
    return np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")


def pixels_equal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Whole-pixel equality across all channels (alpha included)."""
    return np.all(first == second, axis=-1)


def choose_pixels(
    condition: np.ndarray, replacement: np.ndarray, default: np.ndarray
) -> np.ndarray:
    """Select replacement pixels where the rule fires, the center pixel elsewhere."""
    return np.where(condition[..., None], replacement, default)
