from collections.abc import Callable

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.common.neighborhoods import pixels_equal
from scalerack.image_io import ImageInput, as_image_input

CaseTable = list[tuple[np.ndarray, np.ndarray]]
PlaneFunction = Callable[[int, int], np.ndarray]
ExpandFunction = Callable[[np.ndarray], np.ndarray]


@register
def sai2x(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Kreed's 2xSaI edge-aware interpolation."""
    return run_sai(image, expand_sai2x)


@register
def super2xsai(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Kreed's Super 2xSaI, the family's strongest blur."""
    return run_sai(image, expand_super2xsai)


@register
def supereagle(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Kreed's SuperEagle edge detection and blending."""
    return run_sai(image, expand_supereagle)


def run_sai(image: ImageInput, expand: ExpandFunction) -> ImageInput:
    image_input = as_image_input(image)
    return image_input.from_numpy(expand(image_input.rgba()))


def expand_sai2x(values: np.ndarray) -> np.ndarray:
    """Apply the 2xSaI rules once: the top-left output keeps the pixel, the other
    three interpolate toward east, south, and south-east along matched diagonals."""
    plane = extract_planes(values)
    center, east, south, south_east = plane(0, 0), plane(0, 1), plane(1, 0), plane(1, 1)
    north, north_east, north_west, north_east2 = (
        plane(-1, 0),
        plane(-1, 1),
        plane(-1, -1),
        plane(-1, 2),
    )
    west, east2, south_west, south_east2 = plane(0, -1), plane(0, 2), plane(1, -1), plane(1, 2)
    south2_west, south2, south2_east = plane(2, -1), plane(2, 0), plane(2, 1)

    rising = pixels_equal(east, south)
    falling = pixels_equal(center, south_east)
    rising_only = rising & ~falling
    falling_only = falling & ~rising
    both = rising & falling
    neither = ~rising & ~falling

    votes = (
        vote_diagonal(center, east, west, north)
        - vote_diagonal(east, center, east2, north_east)
        - vote_diagonal(east, center, south_west, south2)
        + vote_diagonal(center, east, south_east2, south2_east)
    )

    half_east = (center + east) / 2
    half_south = (center + south) / 2
    quad = (center + east + south + south_east) / 4

    top_keeps_center = (
        pixels_equal(center, south)
        & pixels_equal(center, north_east)
        & ~pixels_equal(east, north)
        & pixels_equal(east, north_east2)
    )
    top_keeps_east = (
        pixels_equal(east, north)
        & pixels_equal(east, south_east)
        & ~pixels_equal(center, north_east)
        & pixels_equal(center, north_west)
    )
    top_right = select_by_case(
        [
            (
                falling_only
                & (
                    (pixels_equal(center, north) & pixels_equal(east, south_east2))
                    | top_keeps_center
                ),
                center,
            ),
            (falling_only, half_east),
            (
                rising_only
                & (
                    (pixels_equal(east, north_east) & pixels_equal(center, south_west))
                    | top_keeps_east
                ),
                east,
            ),
            (rising_only, half_east),
            (both, half_east),
            (neither & top_keeps_center, center),
            (neither & top_keeps_east, east),
            (neither, half_east),
        ]
    )

    left_keeps_center = (
        pixels_equal(center, east)
        & pixels_equal(center, south_west)
        & ~pixels_equal(west, south)
        & pixels_equal(south, south2_west)
    )
    left_keeps_south = (
        pixels_equal(south, west)
        & pixels_equal(south, south_east)
        & ~pixels_equal(center, south_west)
        & pixels_equal(center, north_west)
    )
    bottom_left = select_by_case(
        [
            (
                falling_only
                & (
                    (pixels_equal(center, west) & pixels_equal(south, south2_east))
                    | left_keeps_center
                ),
                center,
            ),
            (falling_only, half_south),
            (
                rising_only
                & (
                    (pixels_equal(south, south_west) & pixels_equal(center, north_east))
                    | left_keeps_south
                ),
                south,
            ),
            (rising_only, half_south),
            (both, half_south),
            (neither & left_keeps_center, center),
            (neither & left_keeps_south, south),
            (neither, half_south),
        ]
    )

    bottom_right = select_by_case(
        [
            (falling_only, center),
            (rising_only, east),
            (both & (votes > 0), center),
            (both & (votes < 0), east),
            (both, quad),
            (neither, quad),
        ]
    )

    return assemble(center, top_right, bottom_left, bottom_right)


def expand_super2xsai(values: np.ndarray) -> np.ndarray:
    """Apply the Super 2xSaI rules once: like 2xSaI but the whole cell shifts
    toward blends, trading crispness for smoothness."""
    plane = extract_planes(values)
    center, east, south, south_east = plane(0, 0), plane(0, 1), plane(1, 0), plane(1, 1)
    north, north_east, north_west, north_east2 = (
        plane(-1, 0),
        plane(-1, 1),
        plane(-1, -1),
        plane(-1, 2),
    )
    west, south_west = plane(0, -1), plane(1, -1)
    south2_west, south2, south2_east, south2_east2 = (
        plane(2, -1),
        plane(2, 0),
        plane(2, 1),
        plane(2, 2),
    )

    rising = pixels_equal(south, east)
    falling = pixels_equal(center, south_east)
    rising_only = rising & ~falling
    falling_only = falling & ~rising
    both = rising & falling
    neither = ~rising & ~falling

    votes = diagonal_votes(plane)

    half_top = (center + east) / 2
    half_bottom = (south + south_east) / 2
    half_vertical = (south + center) / 2

    right_common: CaseTable = [
        (rising_only, south),
        (falling_only, center),
        (both & (votes > 0), east),
        (both & (votes < 0), center),
        (both, half_top),
    ]
    bottom_right = select_by_case(
        [
            *right_common,
            (
                neither
                & pixels_equal(east, south_east)
                & pixels_equal(south_east, south2)
                & ~pixels_equal(south, south2_east)
                & ~pixels_equal(south_east, south2_west),
                toward(south_east, south),
            ),
            (
                neither
                & pixels_equal(center, south)
                & pixels_equal(south, south2_east)
                & ~pixels_equal(south2, south_east)
                & ~pixels_equal(south, south2_east2),
                toward(south, south_east),
            ),
            (neither, half_bottom),
        ]
    )
    top_right = select_by_case(
        [
            *right_common,
            (
                neither
                & pixels_equal(east, south_east)
                & pixels_equal(east, north)
                & ~pixels_equal(center, north_east)
                & ~pixels_equal(east, north_west),
                toward(east, center),
            ),
            (
                neither
                & pixels_equal(center, south)
                & pixels_equal(center, north_east)
                & ~pixels_equal(north, east)
                & ~pixels_equal(center, north_east2),
                toward(center, east),
            ),
            (neither, half_top),
        ]
    )

    bottom_left_blend = (
        falling_only & pixels_equal(west, center) & ~pixels_equal(center, south2_east)
    ) | (
        pixels_equal(center, south_west)
        & pixels_equal(east, center)
        & ~pixels_equal(west, south)
        & ~pixels_equal(center, south2_west)
    )
    bottom_left = select_by_case([(bottom_left_blend, half_vertical), (~bottom_left_blend, south)])

    top_left_blend = (
        rising_only & pixels_equal(south_west, south) & ~pixels_equal(south, north_east)
    ) | (
        pixels_equal(west, south)
        & pixels_equal(south_east, south)
        & ~pixels_equal(south_west, center)
        & ~pixels_equal(south, north_west)
    )
    top_left = select_by_case([(top_left_blend, half_vertical), (~top_left_blend, center)])

    return assemble(top_left, top_right, bottom_left, bottom_right)


def expand_supereagle(values: np.ndarray) -> np.ndarray:
    """Apply the SuperEagle rules once: the cell spanned by a pixel and its
    east/south/south-east neighbors copies or blends toward the matched diagonal."""
    plane = extract_planes(values)
    center, east, west, east2 = plane(0, 0), plane(0, 1), plane(0, -1), plane(0, 2)
    north, north_east = plane(-1, 0), plane(-1, 1)
    south_west, south = plane(1, -1), plane(1, 0)
    south_east, south_east2 = plane(1, 1), plane(1, 2)
    south2, south2_east = plane(2, 0), plane(2, 1)

    rising = pixels_equal(south, east)
    falling = pixels_equal(center, south_east)
    rising_only = rising & ~falling
    falling_only = falling & ~rising
    both = rising & falling
    neither = ~rising & ~falling

    # a matched diagonal blends harder on the side where neighbors extend its edge
    rising_top = rising_only & (pixels_equal(south_west, south) | pixels_equal(east, north_east))
    rising_bottom = rising_only & (pixels_equal(east, east2) | pixels_equal(south, south2))
    falling_top = falling_only & (
        pixels_equal(north, center) | pixels_equal(south_east, south_east2)
    )
    falling_bottom = falling_only & (
        pixels_equal(south_east, south2_east) | pixels_equal(west, center)
    )

    votes = diagonal_votes(plane)
    both_rising = both & (votes > 0)
    both_falling = both & (votes < 0)
    both_tied = both & (votes == 0)

    half_top = (center + east) / 2
    half_bottom = (south + south_east) / 2

    top_left = select_by_case(
        [
            (rising_top, toward(south, center)),
            (rising_only, half_top),
            (falling_only, center),
            (both_rising, half_top),
            (both_falling, center),
            (both_tied, center),
            (neither, corner_mix(center, south, east)),
        ]
    )
    top_right = select_by_case(
        [
            (rising_only, south),
            (falling_top, toward(center, east)),
            (falling_only, half_top),
            (both_rising, south),
            (both_falling, half_top),
            (both_tied, south),
            (neither, corner_mix(east, center, south_east)),
        ]
    )
    bottom_left = select_by_case(
        [
            (rising_only, south),
            (falling_bottom, toward(center, south)),
            (falling_only, half_bottom),
            (both_rising, south),
            (both_falling, half_top),
            (both_tied, south),
            (neither, corner_mix(south, center, south_east)),
        ]
    )
    bottom_right = select_by_case(
        [
            (rising_bottom, toward(south, south_east)),
            (rising_only, half_bottom),
            (falling_only, center),
            (both_rising, half_top),
            (both_falling, center),
            (both_tied, center),
            (neither, corner_mix(south_east, south, east)),
        ]
    )

    return assemble(top_left, top_right, bottom_left, bottom_right)


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


def diagonal_votes(plane: PlaneFunction) -> np.ndarray:
    """Sum the four surrounding votes deciding which diagonal wins a full match."""
    east, center = plane(0, 1), plane(0, 0)
    return (
        vote_diagonal(east, center, plane(1, -1), plane(2, 0))
        + vote_diagonal(east, center, plane(0, -1), plane(-1, 0))
        + vote_diagonal(east, center, plane(2, 1), plane(1, 2))
        + vote_diagonal(east, center, plane(-1, 1), plane(0, 2))
    )


def vote_diagonal(
    first: np.ndarray, second: np.ndarray, probe1: np.ndarray, probe2: np.ndarray
) -> np.ndarray:
    """Vote which diagonal color the two probe pixels side with (Kreed's GetResult)."""
    first_matches1 = pixels_equal(first, probe1)
    first_matches2 = pixels_equal(first, probe2)
    second_matches1 = pixels_equal(second, probe1) & ~first_matches1
    second_matches2 = pixels_equal(second, probe2) & ~first_matches2
    first_score = first_matches1.astype(np.int_) + first_matches2
    second_score = second_matches1.astype(np.int_) + second_matches2
    return (first_score <= 1).astype(np.int_) - (second_score <= 1)


def toward(major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    return (3 * major + minor) / 4


def corner_mix(major: np.ndarray, minor1: np.ndarray, minor2: np.ndarray) -> np.ndarray:
    return (6 * major + minor1 + minor2) / 8


def select_by_case(cases: CaseTable) -> np.ndarray:
    """Pick per pixel from the first matching case; the masks partition the image."""
    conditions = [mask[..., None] for mask, _ in cases]
    choices = [choice for _, choice in cases]
    return np.select(conditions, choices, default=np.nan)


def assemble(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> np.ndarray:
    height, width, channels = top_left.shape
    result = np.empty((height * 2, width * 2, channels), dtype=top_left.dtype)
    result[0::2, 0::2] = top_left
    result[0::2, 1::2] = top_right
    result[1::2, 0::2] = bottom_left
    result[1::2, 1::2] = bottom_right
    return result
