import numpy as np

from scalerack.algorithms.registry import register
from scalerack.common.neighborhoods import PlaneFunction
from scalerack.exceptions import InvalidFactorError
from scalerack.image_io import ImageInput, as_image_input

MINIMUM_FACTOR = 2
MAXIMUM_FACTOR = 6

# xBRZ decision constants
EQUAL_COLOR_TOLERANCE = 30.0
DOMINANT_DIRECTION_THRESHOLD = 3.6
STEEP_DIRECTION_THRESHOLD = 2.2
CENTER_DIAGONAL_WEIGHT = 4

# ITU-R BT.2020 coefficients for the YCbCr color distance
LUMA_RED = 0.2627
LUMA_BLUE = 0.0593
LUMA_GREEN = 1.0 - LUMA_RED - LUMA_BLUE
SCALE_BLUE = 0.5 / (1.0 - LUMA_BLUE)
SCALE_RED = 0.5 / (1.0 - LUMA_RED)

BLEND_NONE = 0
BLEND_NORMAL = 1
BLEND_DOMINANT = 2

# The 3x3 window roles (b, c, d, f, g, h, i around center e) per clockwise
# rotation of the corner under evaluation, as (row, column) offsets.
ROLE_OFFSETS = (
    {"b": (-1, 0), "c": (-1, 1), "d": (0, -1), "f": (0, 1), "g": (1, -1), "h": (1, 0), "i": (1, 1)},
    {
        "b": (0, -1),
        "c": (-1, -1),
        "d": (1, 0),
        "f": (-1, 0),
        "g": (1, 1),
        "h": (0, 1),
        "i": (-1, 1),
    },
    {
        "b": (1, 0),
        "c": (1, -1),
        "d": (0, 1),
        "f": (0, -1),
        "g": (-1, 1),
        "h": (-1, 0),
        "i": (-1, -1),
    },
    {
        "b": (0, 1),
        "c": (1, 1),
        "d": (-1, 0),
        "f": (1, 0),
        "g": (-1, -1),
        "h": (0, -1),
        "i": (1, -1),
    },
)

# Painted cells per line shape as (row, column, numerator, denominator): the
# blend color covers numerator/denominator of the cell (1/1 = full copy).
LINE_SHALLOW = {
    2: ((1, 0, 1, 4), (1, 1, 3, 4)),
    3: ((2, 0, 1, 4), (1, 2, 1, 4), (2, 1, 3, 4), (2, 2, 1, 1)),
    4: ((3, 0, 1, 4), (2, 2, 1, 4), (3, 1, 3, 4), (2, 3, 3, 4), (3, 2, 1, 1), (3, 3, 1, 1)),
    5: (
        (4, 0, 1, 4),
        (3, 2, 1, 4),
        (2, 4, 1, 4),
        (4, 1, 3, 4),
        (3, 3, 3, 4),
        (4, 2, 1, 1),
        (4, 3, 1, 1),
        (4, 4, 1, 1),
        (3, 4, 1, 1),
    ),
    6: (
        (5, 0, 1, 4),
        (4, 2, 1, 4),
        (3, 4, 1, 4),
        (5, 1, 3, 4),
        (4, 3, 3, 4),
        (3, 5, 3, 4),
        (5, 2, 1, 1),
        (5, 3, 1, 1),
        (5, 4, 1, 1),
        (5, 5, 1, 1),
        (4, 4, 1, 1),
        (4, 5, 1, 1),
    ),
}
LINE_STEEP = {
    2: ((0, 1, 1, 4), (1, 1, 3, 4)),
    3: ((0, 2, 1, 4), (2, 1, 1, 4), (1, 2, 3, 4), (2, 2, 1, 1)),
    4: ((0, 3, 1, 4), (2, 2, 1, 4), (1, 3, 3, 4), (3, 2, 3, 4), (2, 3, 1, 1), (3, 3, 1, 1)),
    5: (
        (0, 4, 1, 4),
        (2, 3, 1, 4),
        (4, 2, 1, 4),
        (1, 4, 3, 4),
        (3, 3, 3, 4),
        (2, 4, 1, 1),
        (3, 4, 1, 1),
        (4, 4, 1, 1),
        (4, 3, 1, 1),
    ),
    6: (
        (0, 5, 1, 4),
        (2, 4, 1, 4),
        (4, 3, 1, 4),
        (1, 5, 3, 4),
        (3, 4, 3, 4),
        (5, 3, 3, 4),
        (2, 5, 1, 1),
        (3, 5, 1, 1),
        (4, 5, 1, 1),
        (5, 5, 1, 1),
        (4, 4, 1, 1),
        (5, 4, 1, 1),
    ),
}
LINE_STEEP_AND_SHALLOW = {
    2: ((1, 0, 1, 4), (0, 1, 1, 4), (1, 1, 5, 6)),
    3: ((2, 0, 1, 4), (0, 2, 1, 4), (2, 1, 3, 4), (1, 2, 3, 4), (2, 2, 1, 1)),
    4: (
        (3, 1, 3, 4),
        (1, 3, 3, 4),
        (3, 0, 1, 4),
        (0, 3, 1, 4),
        (2, 2, 1, 3),
        (3, 3, 1, 1),
        (3, 2, 1, 1),
        (2, 3, 1, 1),
    ),
    5: (
        (0, 4, 1, 4),
        (2, 3, 1, 4),
        (1, 4, 3, 4),
        (4, 0, 1, 4),
        (3, 2, 1, 4),
        (4, 1, 3, 4),
        (3, 3, 2, 3),
        (2, 4, 1, 1),
        (3, 4, 1, 1),
        (4, 4, 1, 1),
        (4, 2, 1, 1),
        (4, 3, 1, 1),
    ),
    6: (
        (0, 5, 1, 4),
        (2, 4, 1, 4),
        (1, 5, 3, 4),
        (3, 4, 3, 4),
        (5, 0, 1, 4),
        (4, 2, 1, 4),
        (5, 1, 3, 4),
        (4, 3, 3, 4),
        (2, 5, 1, 1),
        (3, 5, 1, 1),
        (4, 5, 1, 1),
        (5, 5, 1, 1),
        (4, 4, 1, 1),
        (5, 4, 1, 1),
        (5, 2, 1, 1),
        (5, 3, 1, 1),
    ),
}
LINE_DIAGONAL = {
    2: ((1, 1, 1, 2),),
    3: ((1, 2, 1, 8), (2, 1, 1, 8), (2, 2, 7, 8)),
    4: ((3, 2, 1, 2), (2, 3, 1, 2), (3, 3, 1, 1)),
    5: ((4, 2, 1, 8), (3, 3, 1, 8), (2, 4, 1, 8), (4, 3, 7, 8), (3, 4, 7, 8), (4, 4, 1, 1)),
    6: ((5, 3, 1, 2), (4, 4, 1, 2), (3, 5, 1, 2), (4, 5, 1, 1), (5, 5, 1, 1), (5, 4, 1, 1)),
}
CORNER = {
    2: ((1, 1, 21, 100),),
    3: ((2, 2, 45, 100),),
    4: ((3, 3, 68, 100), (3, 2, 9, 100), (2, 3, 9, 100)),
    5: ((4, 4, 86, 100), (4, 3, 23, 100), (3, 4, 23, 100)),
    6: ((5, 5, 97, 100), (4, 5, 42, 100), (5, 4, 42, 100), (5, 3, 6, 100), (3, 5, 6, 100)),
}


@register
def xbrz(image: ImageInput, factor: int = 2) -> ImageInput:
    """Enlarge pixel art 2x-6x with Zenju's xBRZ edge-slope detection and blending.

    Args:
        factor: Integer scale factor between 2 and 6.
    """
    if float(factor) not in {float(value) for value in range(MINIMUM_FACTOR, MAXIMUM_FACTOR + 1)}:
        raise InvalidFactorError(
            f"xbrz requires an integer factor between {MINIMUM_FACTOR} and {MAXIMUM_FACTOR}, "
            f"got {factor!r}"
        )
    image_input = as_image_input(image)
    original = image_input.numpy()
    # images with an alpha channel treat out-of-bounds pixels as transparent
    # black, opaque images duplicate their borders
    transparent_border = original.ndim == 3 and original.shape[2] == 4
    result = expand_xbrz(image_input.rgba(), int(factor), transparent_border)
    return image_input.from_numpy(result)


def expand_xbrz(values: np.ndarray, factor: int, transparent_border: bool) -> np.ndarray:
    """Blend the four corners of every pixel's output block along detected edges."""
    quantized = np.rint(values.astype(np.float64) * 255.0)
    height, width = quantized.shape[:2]
    padded = np.pad(
        quantized,
        ((2, 2), (2, 2), (0, 0)),
        mode="constant" if transparent_border else "edge",
    )

    def corner_plane(row_offset: int, column_offset: int) -> np.ndarray:
        # views over the corner-anchor grid, which starts one pixel up-left of
        # the image so border pixels get real top/left corner decisions
        return padded[
            1 + row_offset : 1 + row_offset + height + 1,
            1 + column_offset : 1 + column_offset + width + 1,
        ]

    def plane(row_offset: int, column_offset: int) -> np.ndarray:
        return padded[
            2 + row_offset : 2 + row_offset + height,
            2 + column_offset : 2 + column_offset + width,
        ]

    corners = blend_corners(corner_plane)
    cells = np.broadcast_to(
        quantized, (factor, factor, height, width, 4)
    ).copy()  # output block per pixel, prefilled with the center color

    for rotation in range(4):
        blend_pixels(cells, plane, corners, rotation, factor)

    result = np.empty((height * factor, width * factor, 4), dtype=np.float32)
    for row in range(factor):
        for column in range(factor):
            result[row::factor, column::factor] = cells[row, column] / 255.0
    return result


def blend_corners(plane: PlaneFunction) -> list[np.ndarray]:
    """Preprocess every block corner into a blend strength, evaluated on the
    corner-anchor grid and sliced back per pixel, ordered like the rotations:
    bottom-right, top-right, top-left, bottom-left."""
    center, east, south, south_east = plane(0, 0), plane(0, 1), plane(1, 0), plane(1, 1)

    # skip corners fully spanned by two equal-colored lines
    skip = (pixels_match(center, east) & pixels_match(south, south_east)) | (
        pixels_match(center, south) & pixels_match(east, south_east)
    )

    falling = (  # strength of the center/south-east diagonal
        color_distance(plane(1, -1), center)
        + color_distance(center, plane(-1, 1))
        + color_distance(plane(2, 0), south_east)
        + color_distance(south_east, plane(0, 2))
        + CENTER_DIAGONAL_WEIGHT * color_distance(south, east)
    )
    rising = (  # strength of the east/south diagonal ("fk")
        color_distance(plane(0, -1), south)
        + color_distance(south, plane(2, 1))
        + color_distance(plane(-1, 0), east)
        + color_distance(east, plane(1, 2))
        + CENTER_DIAGONAL_WEIGHT * color_distance(center, south_east)
    )

    def strength(
        winner: np.ndarray,
        loser: np.ndarray,
        pixel: np.ndarray,
        adjacent: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        wins = (
            ~skip
            & (winner < loser)
            & ~pixels_match(pixel, adjacent[0])
            & ~pixels_match(pixel, adjacent[1])
        )
        dominant = DOMINANT_DIRECTION_THRESHOLD * winner < loser
        return np.where(wins, np.where(dominant, BLEND_DOMINANT, BLEND_NORMAL), BLEND_NONE)

    corner_center = strength(falling, rising, center, (east, south))
    corner_south_east = strength(falling, rising, south_east, (south, east))
    corner_south = strength(rising, falling, south, (center, south_east))
    corner_east = strength(rising, falling, east, (center, south_east))

    # anchor (y, x) decides the corner between pixels (y, x) and (y+1, x+1);
    # slice each pixel's four corners from the surrounding anchors
    bottom_right = corner_center[1:, 1:]
    top_right = corner_south[:-1, 1:]
    top_left = corner_south_east[:-1, :-1]
    bottom_left = corner_east[1:, :-1]
    return [bottom_right, top_right, top_left, bottom_left]


def blend_pixels(
    cells: np.ndarray,
    plane: PlaneFunction,
    corners: list[np.ndarray],
    rotation: int,
    factor: int,
) -> None:
    """Blend one corner (selected by rotation) of every pixel's output block."""
    corner = corners[rotation]
    active = corner >= BLEND_NORMAL
    if not active.any():
        return

    roles = {name: plane(*offset) for name, offset in ROLE_OFFSETS[rotation].items()}
    center = plane(0, 0)
    # adjacent corners in the rotated frame, for the line-blend admissibility checks
    clockwise = corners[(rotation + 1) % 4]
    counter_clockwise = corners[(rotation - 1) % 4]

    blocked = (
        ((clockwise != BLEND_NONE) & ~colors_close(center, roles["g"]))
        | ((counter_clockwise != BLEND_NONE) & ~colors_close(center, roles["c"]))
        | (  # L-shape: blend the corner only
            ~colors_close(center, roles["i"])
            & colors_close(roles["g"], roles["h"])
            & colors_close(roles["h"], roles["i"])
            & colors_close(roles["i"], roles["f"])
            & colors_close(roles["f"], roles["c"])
        )
    )
    line_blend = active & ((corner >= BLEND_DOMINANT) | ~blocked)

    blend_color = np.where(
        (color_distance(center, roles["f"]) <= color_distance(center, roles["h"]))[..., None],
        roles["f"],
        roles["h"],
    )

    edge_shallow = color_distance(roles["f"], roles["g"])
    edge_steep = color_distance(roles["h"], roles["c"])
    shallow = (
        (STEEP_DIRECTION_THRESHOLD * edge_shallow <= edge_steep)
        & ~pixels_match(center, roles["g"])
        & ~pixels_match(roles["d"], roles["g"])
    )
    steep = (
        (STEEP_DIRECTION_THRESHOLD * edge_steep <= edge_shallow)
        & ~pixels_match(center, roles["c"])
        & ~pixels_match(roles["b"], roles["c"])
    )

    shapes = (
        (line_blend & shallow & steep, LINE_STEEP_AND_SHALLOW[factor]),
        (line_blend & shallow & ~steep, LINE_SHALLOW[factor]),
        (line_blend & ~shallow & steep, LINE_STEEP[factor]),
        (line_blend & ~shallow & ~steep, LINE_DIAGONAL[factor]),
        (active & ~line_blend, CORNER[factor]),
    )
    for mask, painted_cells in shapes:
        if not mask.any():
            continue
        for row, column, numerator, denominator in painted_cells:
            target_row, target_column = rotate_cell(row, column, rotation, factor)
            cell = cells[target_row, target_column]
            cell[mask] = alpha_gradient(blend_color[mask], cell[mask], numerator, denominator)


def rotate_cell(row: int, column: int, rotation: int, factor: int) -> tuple[int, int]:
    """Map a cell of the rotated output block back to the unrotated block."""
    for _ in range(rotation):
        row, column = factor - 1 - column, row
    return row, column


def alpha_gradient(
    front: np.ndarray, back: np.ndarray, numerator: int, denominator: int
) -> np.ndarray:
    """Cover ``numerator/denominator`` of the back color with the front color,
    weighting color contributions by each side's alpha."""
    if numerator == denominator:
        return front
    weight_front = front[..., 3] * numerator
    weight_back = back[..., 3] * (denominator - numerator)
    weight_sum = weight_front + weight_back
    alpha = weight_sum / denominator
    scale = np.divide(1.0, weight_sum, out=np.zeros_like(weight_sum), where=weight_sum > 0)
    rgb = (front[..., :3] * weight_front[..., None] + back[..., :3] * weight_back[..., None]) * (
        scale[..., None]
    )
    return np.concatenate((rgb, alpha[..., None]), axis=-1)


def color_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Alpha-aware color distance: YCbCr distance of the RGB difference,
    scaled by the smaller alpha plus the alpha difference."""
    difference = quantize_difference(first[..., :3] - second[..., :3])
    red, green, blue = difference[..., 0], difference[..., 1], difference[..., 2]
    luma = LUMA_RED * red + LUMA_GREEN * green + LUMA_BLUE * blue
    chroma_blue = SCALE_BLUE * (blue - luma)
    chroma_red = SCALE_RED * (red - luma)
    base = np.sqrt(luma * luma + chroma_blue * chroma_blue + chroma_red * chroma_red)

    alpha_low = np.minimum(first[..., 3], second[..., 3]) / 255.0
    alpha_high = np.maximum(first[..., 3], second[..., 3]) / 255.0
    return alpha_low * base + 255.0 * (alpha_high - alpha_low)


def quantize_difference(difference: np.ndarray) -> np.ndarray:
    """Halve the channel-difference precision, matching xBRZ's precomputed
    distance table so decisions agree with it bit for bit."""
    return np.trunc(difference / 2.0) * 2.0


def colors_close(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return color_distance(first, second) < EQUAL_COLOR_TOLERANCE


def pixels_match(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Exact whole-pixel equality (all channels, alpha included)."""
    return np.all(first == second, axis=-1)
