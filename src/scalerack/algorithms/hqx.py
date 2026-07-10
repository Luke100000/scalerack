import numpy as np

from scalerack.algorithms.hqx_tables import (
    HQ2X_INDEX,
    HQ2X_SPECS,
    HQ3X_INDEX,
    HQ3X_SPECS,
    HQ4X_INDEX,
    HQ4X_SPECS,
)
from scalerack.algorithms.registry import register
from scalerack.common.neighborhoods import extract_planes
from scalerack.image_io import ImageInput, as_image_input

Branch = tuple[int, tuple[int, ...]]
CellRule = tuple[int | tuple[int, int], Branch, Branch]
CellSpecs = tuple[tuple[CellRule, ...], ...]
CellIndex = tuple[bytes, ...]

# Row-major 3x3 window around the center pixel
NEIGHBOR_OFFSETS = {
    1: (-1, -1),
    2: (-1, 0),
    3: (-1, 1),
    4: (0, -1),
    5: (0, 0),
    6: (0, 1),
    7: (1, -1),
    8: (1, 0),
    9: (1, 1),
}
PATTERN_BITS = {1: 1, 2: 2, 3: 4, 4: 8, 6: 16, 7: 32, 8: 64, 9: 128}

# Perceptual similarity thresholds on the 0-255 YUV scale
LUMA_THRESHOLD = 48
U_THRESHOLD = 7
V_THRESHOLD = 6

# Blend weights
INTERPOLATION_WEIGHTS = {
    0: (1,),
    1: (3, 1),
    2: (2, 1, 1),
    3: (7, 1),
    4: (2, 7, 7),
    5: (1, 1),
    6: (5, 2, 1),
    7: (6, 1, 1),
    8: (5, 3),
    9: (2, 3, 3),
    10: (14, 1, 1),
}


@register
def hq2x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Maxim Stepin's hq2x pattern interpolation."""
    return run_hqx(image, 2, HQ2X_SPECS, HQ2X_INDEX)


@register
def hq3x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 3x with Maxim Stepin's hq3x pattern interpolation."""
    return run_hqx(image, 3, HQ3X_SPECS, HQ3X_INDEX)


@register
def hq4x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 4x with Maxim Stepin's hq4x pattern interpolation."""
    return run_hqx(image, 4, HQ4X_SPECS, HQ4X_INDEX)


def run_hqx(image: ImageInput, factor: int, specs: CellSpecs, index: CellIndex) -> ImageInput:
    image_input = as_image_input(image)
    return image_input.from_numpy(expand_hqx(image_input.rgba(), factor, specs, index))


def expand_hqx(values: np.ndarray, factor: int, specs: CellSpecs, index: CellIndex) -> np.ndarray:
    """Classify each 3x3 neighborhood into one of 256 patterns and paint the
    output block from that pattern's blend rules."""
    plane = extract_planes(values)
    neighbors = {label: plane(*offset) for label, offset in NEIGHBOR_OFFSETS.items()}
    yuv = {label: rgb_to_yuv(neighbor) for label, neighbor in neighbors.items()}

    pattern = np.zeros(values.shape[:2], dtype=np.uint8)
    for label, bit in PATTERN_BITS.items():
        pattern |= np.where(yuv_differs(yuv[5], yuv[label]), np.uint8(bit), np.uint8(0))
    condition_pairs = {
        rule[0] for cell_rules in specs for rule in cell_rules if isinstance(rule[0], tuple)
    }
    condition_masks = {pair: yuv_differs(yuv[pair[0]], yuv[pair[1]]) for pair in condition_pairs}

    height, width, channels = values.shape
    result = np.empty((height * factor, width * factor, channels), dtype=values.dtype)
    for cell, (cell_specs, cell_index) in enumerate(zip(specs, index, strict=True)):
        row, column = divmod(cell, factor)
        rule_map = np.frombuffer(cell_index, dtype=np.uint8)[pattern]
        cell_values = np.empty_like(values)
        for rule_id, (condition, when_true, when_false) in enumerate(cell_specs):
            rule_mask = rule_map == rule_id
            if not rule_mask.any():
                continue
            if isinstance(condition, tuple):
                fires = condition_masks[condition]
                blend_into(cell_values, rule_mask & fires, when_true, neighbors)
                blend_into(cell_values, rule_mask & ~fires, when_false, neighbors)
            else:
                blend_into(cell_values, rule_mask, when_true, neighbors)
        result[row::factor, column::factor] = cell_values
    return result


def blend_into(
    output: np.ndarray,
    mask: np.ndarray,
    branch: Branch,
    neighbors: dict[int, np.ndarray],
) -> None:
    """Write the branch's weighted neighbor blend into the masked pixels."""
    if not mask.any():
        return
    interpolation, labels = branch
    weights = INTERPOLATION_WEIGHTS[interpolation]
    blended = sum(
        weight * neighbors[label][mask] for weight, label in zip(weights, labels, strict=True)
    ) / sum(weights)
    output[mask] = blended


def rgb_to_yuv(values: np.ndarray) -> np.ndarray:
    rgb = np.rint(values[..., :3].astype(np.float64) * 255.0)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = np.trunc(0.299 * red + 0.587 * green + 0.114 * blue)
    u_chroma = np.trunc(-0.169 * red - 0.331 * green + 0.5 * blue) + 128
    v_chroma = np.trunc(0.5 * red - 0.419 * green - 0.081 * blue) + 128
    return np.stack((luma, u_chroma, v_chroma), axis=-1)


def yuv_differs(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first - second)
    return (
        (difference[..., 0] > LUMA_THRESHOLD)
        | (difference[..., 1] > U_THRESHOLD)
        | (difference[..., 2] > V_THRESHOLD)
    )
