from collections.abc import Callable

import numpy as np

from scalerack.common.resample import drop_channel_axis, ensure_channel_axis
from scalerack.image_io import ImageInput, as_image_input

ExpandFunction = Callable[[np.ndarray], np.ndarray]
PlaneFunction = Callable[[int, int], np.ndarray]


def run_expansion(image: ImageInput, expand: ExpandFunction) -> ImageInput:
    """Run a fixed-factor block expansion while preserving the input format."""
    image_input = as_image_input(image)
    array = image_input.numpy()
    values = ensure_channel_axis(array)
    result = drop_channel_axis(expand(values), array.ndim)
    return image_input.from_numpy(result)


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


def extract_planes(values: np.ndarray) -> PlaneFunction:
    """Return an accessor for the padded neighborhood, offset in (row, column)."""
    height, width = values.shape[:2]
    padded = np.pad(values, ((2, 2), (2, 2), (0, 0)), mode="edge")

    def plane(row_offset: int, column_offset: int) -> np.ndarray:
        return padded[
            2 + row_offset : 2 + row_offset + height,
            2 + column_offset : 2 + column_offset + width,
        ]

    return plane


def pixels_equal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Whole-pixel equality across all channels (alpha included)."""
    return np.all(first == second, axis=-1)


def choose_pixels(
    condition: np.ndarray, replacement: np.ndarray, default: np.ndarray
) -> np.ndarray:
    """Select replacement pixels where the rule fires, the center pixel elsewhere."""
    return np.where(condition[..., None], replacement, default)
