import numpy as np

from scalerack.algorithms.registry import register
from scalerack.common.neighborhoods import pixels_equal
from scalerack.image_io import ImageInput, as_image_input

CaseTable = list[tuple[np.ndarray, np.ndarray]]


@register
def supereagle(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Kreed's SuperEagle edge detection and blending."""
    image_input = as_image_input(image)
    return image_input.from_numpy(expand_supereagle(image_input.rgba()))


def expand_supereagle(values: np.ndarray) -> np.ndarray:
    """Apply the SuperEagle rules once: the cell spanned by a pixel and its
    east/south/south-east neighbors copies or blends toward the matched diagonal."""
    height, width, channels = values.shape
    padded = np.pad(values, ((2, 2), (2, 2), (0, 0)), mode="edge")

    def plane(row_offset: int, column_offset: int) -> np.ndarray:
        return padded[
            2 + row_offset : 2 + row_offset + height,
            2 + column_offset : 2 + column_offset + width,
        ]

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

    votes = (
        vote_diagonal(east, center, south_west, south2)
        + vote_diagonal(east, center, west, north)
        + vote_diagonal(east, center, south2_east, south_east2)
        + vote_diagonal(east, center, north_east, east2)
    )
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

    result = np.empty((height * 2, width * 2, channels), dtype=values.dtype)
    result[0::2, 0::2] = top_left
    result[0::2, 1::2] = top_right
    result[1::2, 0::2] = bottom_left
    result[1::2, 1::2] = bottom_right
    return result


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
