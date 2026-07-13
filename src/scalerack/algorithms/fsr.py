"""AMD FidelityFX Super Resolution 1.0: EASU upscaling plus RCAS sharpening.

Port of the non-packed 32-bit reference passes ``FsrEasuF`` and ``FsrRcasF``
from ``ffx_fsr1.h`` (with the approximate-arithmetic helpers from ``ffx_a.h``),
Copyright (c) 2021 Advanced Micro Devices, Inc., MIT license
(https://github.com/GPUOpen-Effects/FidelityFX-FSR,
commit a21ffb8f6c13233ba336352bdff293894c706575).

Deviations from the reference, both stills-library conventions: the alpha
channel rides through EASU with the color-derived weights and dering clamp and
passes through RCAS unchanged, and the reference's optional compile-time
``FSR_RCAS_DENOISE`` term (a temporal-grain guard for video) is omitted.
"""

import math
import numbers

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.exceptions import InvalidFactorError
from scalerack.image_io import ImageInput, as_image_input

RCAS_DEFAULT_SHARPNESS = 0.2

# the reference's hard bound on the negative lobe ("limit of providing
# unnatural results for sharpening")
RCAS_LIMIT = 0.25 - 1.0 / 16.0

# integer magics of the reference's approximate reciprocal / inverse-sqrt
# (APrxLoRcpF1, APrxMedRcpF1, APrxLoRsqF1 in ffx_a.h)
LOW_RECIPROCAL_BITS = np.uint32(0x7EF07EBB)
MEDIUM_RECIPROCAL_BITS = np.uint32(0x7EF19FFF)
LOW_INVERSE_SQRT_BITS = np.uint32(0x5F347D74)

# below this squared gradient magnitude the direction collapses to horizontal
DIRECTION_EPSILON = 1.0 / 32768.0

# polynomial window of the EASU kernel: (25/16*(2/5*d2-1)^2-(25/16-1)) * (lobe*d2-1)^2
KERNEL_BASE_SLOPE = 2.0 / 5.0
KERNEL_BASE_SCALE = 25.0 / 16.0
# negative-lobe strength: LOBE_FLAT on flat areas, shifted by LOBE_EDGE_SHIFT on edges
LOBE_FLAT = 0.5
LOBE_EDGE_SHIFT = (1.0 / 4.0 - 0.04) - 0.5

# EASU tap layout around the quad base (column offset, row offset):
#     b c
#   e f g h
#   i j k l
#     n o
TAP_OFFSETS = {
    "b": (0, -1),
    "c": (1, -1),
    "e": (-1, 0),
    "f": (0, 0),
    "g": (1, 0),
    "h": (2, 0),
    "i": (-1, 1),
    "j": (0, 1),
    "k": (1, 1),
    "l": (2, 1),
    "n": (0, 2),
    "o": (1, 2),
}
# accumulation order of FsrEasuF's FsrEasuTapF calls (float summation order matters)
TAP_ORDER = ("b", "c", "i", "j", "f", "e", "k", "l", "h", "g", "o", "n")
# per quad pixel: its cross neighborhood (above, left, center, right, below)
# in FsrEasuSetF call order; the bilinear corner weight follows this order too
ANALYSIS_NEIGHBORHOODS = (
    ("f", ("b", "e", "f", "g", "j")),
    ("g", ("c", "f", "g", "h", "k")),
    ("j", ("f", "i", "j", "k", "n")),
    ("k", ("g", "j", "k", "l", "o")),
)
QUAD_TAPS = ("f", "g", "j", "k")


@register()
def fsr(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    sharpness: float | None = RCAS_DEFAULT_SHARPNESS,
) -> ImageInput:
    """Upscale with AMD FSR 1: EASU edge-adaptive upsampling plus RCAS sharpening.

    Designed for continuous-tone content; AMD's published quality presets span
    factors of about 1.3x to 2x per axis (larger factors work but soften).
    Upscaling only; a factor of exactly 1 applies pure RCAS sharpening.

    Args:
        sharpness: RCAS strength in stops of attenuation: 0.0 is maximum
            sharpening and each additional stop halves the effect; ``None``
            disables the pass entirely (EASU-only output).
    """
    image_input = as_image_input(image)
    validate_sharpness(sharpness)
    output_width, output_height = image_input.get_target_dimensions(width, height, factor)
    values = image_input.rgba()
    input_height, input_width = values.shape[:2]
    if output_width < input_width or output_height < input_height:
        raise InvalidFactorError(
            f"fsr upscales only: requested {output_width}x{output_height} is smaller "
            f"than the {input_width}x{input_height} input"
        )
    if (output_height, output_width) != (input_height, input_width):
        values = apply_easu(values, output_height, output_width)
    if sharpness is not None:
        values = apply_rcas(values, sharpness)
    return image_input.from_numpy(values)


def validate_sharpness(sharpness: float | None) -> None:
    """Reject sharpness values outside the published scale (None disables RCAS)."""
    if sharpness is None:
        return
    if isinstance(sharpness, bool) or not isinstance(sharpness, numbers.Real):
        raise InvalidFactorError(
            f"sharpness must be None or a non-negative number of stops, got {sharpness!r}"
        )
    value = float(sharpness)
    if math.isnan(value) or value < 0:
        raise InvalidFactorError(
            f"sharpness must be None or a non-negative number of stops, got {sharpness!r}"
        )


def apply_easu(values: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    """Reference FsrEasuF over a float32 RGBA image in [0, 1], fully vectorized."""
    input_height, input_width = values.shape[:2]
    one = np.float32(1.0)
    half = np.float32(0.5)

    # FsrEasuCon, kept in float32 like the GPU constant setup
    scale_x = np.float32(input_width) * (one / np.float32(output_width))
    scale_y = np.float32(input_height) * (one / np.float32(output_height))
    offset_x = half * np.float32(input_width) * (one / np.float32(output_width)) - half
    offset_y = half * np.float32(input_height) * (one / np.float32(output_height)) - half

    column_positions = np.arange(output_width, dtype=np.float32) * scale_x + offset_x
    row_positions = np.arange(output_height, dtype=np.float32) * scale_y + offset_y
    column_base = np.floor(column_positions)
    row_base = np.floor(row_positions)
    frac_x = column_positions - column_base
    frac_y = row_positions - row_base
    column_index = column_base.astype(np.int64)
    row_index = row_base.astype(np.int64)

    def gather(grid: np.ndarray, letter: str) -> np.ndarray:
        dx, dy = TAP_OFFSETS[letter]
        rows = np.clip(row_index + dy, 0, input_height - 1)[:, None]
        columns = np.clip(column_index + dx, 0, input_width - 1)[None, :]
        return grid[rows, columns]

    # the reference's "luma times 2": B*0.5 + (R*0.5 + G)
    luma_grid = values[:, :, 2] * half + (values[:, :, 0] * half + values[:, :, 1])
    tap_luma = {letter: gather(luma_grid, letter) for letter in TAP_OFFSETS}

    # bilinear corner weights of the sub-pixel position, one per quad pixel
    corner_weights = {
        "f": (one - frac_x)[None, :] * (one - frac_y)[:, None],
        "g": frac_x[None, :] * (one - frac_y)[:, None],
        "j": (one - frac_x)[None, :] * frac_y[:, None],
        "k": frac_x[None, :] * frac_y[:, None],
    }

    # FsrEasuSetF: accumulate direction and edge length over the four quad pixels
    direction_x = np.zeros((output_height, output_width), dtype=np.float32)
    direction_y = np.zeros((output_height, output_width), dtype=np.float32)
    length = np.zeros((output_height, output_width), dtype=np.float32)
    for quad_letter, (above, left, center, right, below) in ANALYSIS_NEIGHBORHOODS:
        weight = corner_weights[quad_letter]
        luma_above = tap_luma[above]
        luma_left = tap_luma[left]
        luma_center = tap_luma[center]
        luma_right = tap_luma[right]
        luma_below = tap_luma[below]

        contrast_x = np.maximum(np.abs(luma_right - luma_center), np.abs(luma_center - luma_left))
        inverse_contrast_x = approx_low_reciprocal(contrast_x)
        gradient_x = luma_right - luma_left
        direction_x += gradient_x * weight
        edge_x = saturate(np.abs(gradient_x) * inverse_contrast_x)
        length += (edge_x * edge_x) * weight

        contrast_y = np.maximum(np.abs(luma_below - luma_center), np.abs(luma_center - luma_above))
        inverse_contrast_y = approx_low_reciprocal(contrast_y)
        gradient_y = luma_below - luma_above
        direction_y += gradient_y * weight
        edge_y = saturate(np.abs(gradient_y) * inverse_contrast_y)
        length += (edge_y * edge_y) * weight

    # normalize the direction, collapsing near-zero gradients to horizontal
    direction_squared = direction_x * direction_x + direction_y * direction_y
    near_zero = direction_squared < np.float32(DIRECTION_EPSILON)
    inverse_norm = approx_low_inverse_sqrt(direction_squared)
    inverse_norm = np.where(near_zero, one, inverse_norm)
    direction_x = np.where(near_zero, one, direction_x)
    direction_x = direction_x * inverse_norm
    direction_y = direction_y * inverse_norm

    # shape the edge amount and derive the anisotropic kernel parameters
    edge_amount = length * half
    edge_amount = edge_amount * edge_amount
    stretch = (direction_x * direction_x + direction_y * direction_y) * approx_low_reciprocal(
        np.maximum(np.abs(direction_x), np.abs(direction_y))
    )
    kernel_scale_x = one + (stretch - one) * edge_amount
    kernel_scale_y = one + np.float32(-0.5) * edge_amount
    lobe = np.float32(LOBE_FLAT) + np.float32(LOBE_EDGE_SHIFT) * edge_amount
    clip_point = approx_low_reciprocal(lobe)

    # dering bounds: min/max of the four nearest source pixels
    quad_colors = {letter: gather(values, letter) for letter in QUAD_TAPS}
    lower_bound = np.minimum(
        np.minimum(quad_colors["f"], np.minimum(quad_colors["g"], quad_colors["j"])),
        quad_colors["k"],
    )
    upper_bound = np.maximum(
        np.maximum(quad_colors["f"], np.maximum(quad_colors["g"], quad_colors["j"])),
        quad_colors["k"],
    )

    # FsrEasuTapF: accumulate the 12 taps through the polynomial window
    color_sum = np.zeros((output_height, output_width, 4), dtype=np.float32)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float32)
    for letter in TAP_ORDER:
        dx, dy = TAP_OFFSETS[letter]
        offset_to_tap_x = (np.float32(dx) - frac_x)[None, :]
        offset_to_tap_y = (np.float32(dy) - frac_y)[:, None]
        rotated_x = offset_to_tap_x * direction_x + offset_to_tap_y * direction_y
        rotated_y = offset_to_tap_x * (-direction_y) + offset_to_tap_y * direction_x
        rotated_x = rotated_x * kernel_scale_x
        rotated_y = rotated_y * kernel_scale_y
        distance_squared = rotated_x * rotated_x + rotated_y * rotated_y
        distance_squared = np.minimum(distance_squared, clip_point)
        base = np.float32(KERNEL_BASE_SLOPE) * distance_squared - one
        window = lobe * distance_squared - one
        base = base * base
        window = window * window
        base = np.float32(KERNEL_BASE_SCALE) * base - np.float32(KERNEL_BASE_SCALE - 1.0)
        weight = base * window
        color = quad_colors[letter] if letter in quad_colors else gather(values, letter)
        color_sum += color * weight[:, :, None]
        weight_sum += weight

    with np.errstate(divide="ignore", invalid="ignore"):
        normalizer = one / weight_sum
    filtered = color_sum * normalizer[:, :, None]
    # fmin/fmax mirror GPU min/max, which drop a NaN operand
    return np.fmin(upper_bound, np.fmax(lower_bound, filtered))


def apply_rcas(values: np.ndarray, sharpness: float) -> np.ndarray:
    """Reference FsrRcasF over a float32 RGBA image in [0, 1]; alpha passes through."""
    one = np.float32(1.0)
    four = np.float32(4.0)
    attenuation = np.exp2(np.float32(-float(sharpness)))

    padded = np.pad(values[:, :, :3], ((1, 1), (1, 1), (0, 0)), mode="edge")
    north = padded[:-2, 1:-1]
    west = padded[1:-1, :-2]
    center = padded[1:-1, 1:-1]
    east = padded[1:-1, 2:]
    south = padded[2:, 1:-1]

    ring_min = np.minimum(np.minimum(north, np.minimum(west, east)), south)
    ring_max = np.maximum(np.maximum(north, np.maximum(west, east)), south)

    # solve for the strongest negative lobe that cannot clip out of [0, 1];
    # degenerate rings divide by zero exactly like the GPU (min/max drop the NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_min = np.minimum(ring_min, center) * (one / (four * ring_max))
        hit_max = (one - np.maximum(ring_max, center)) * (one / (four * ring_min - four))
    lobe_per_channel = np.fmax(-hit_min, hit_max)
    lobe = (
        np.fmax(
            np.float32(-RCAS_LIMIT),
            np.fmin(
                np.fmax(
                    lobe_per_channel[:, :, 0],
                    np.fmax(lobe_per_channel[:, :, 1], lobe_per_channel[:, :, 2]),
                ),
                np.float32(0.0),
            ),
        )
        * attenuation
    )

    blend_normalizer = approx_medium_reciprocal(four * lobe + one)
    lobe3 = lobe[:, :, None]
    # summation order matches the reference: north, west, south, east, center
    sharpened = (
        lobe3 * north + lobe3 * west + lobe3 * south + lobe3 * east + center
    ) * blend_normalizer[:, :, None]

    result = values.copy()
    # the GPU stores to a UNORM target, which saturates; degenerate neighborhoods
    # (e.g. a bright pixel in a uniform extreme ring) can resolve past the range
    result[:, :, :3] = saturate(sharpened)
    return result


def saturate(values: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(values, np.float32(0.0)), np.float32(1.0))


def approx_low_reciprocal(values: np.ndarray) -> np.ndarray:
    """The reference's APrxLoRcpF1 integer-magic reciprocal."""
    return (LOW_RECIPROCAL_BITS - values.view(np.uint32)).view(np.float32)


def approx_medium_reciprocal(values: np.ndarray) -> np.ndarray:
    """The reference's APrxMedRcpF1: one Newton step on the integer-magic estimate."""
    estimate = (MEDIUM_RECIPROCAL_BITS - values.view(np.uint32)).view(np.float32)
    return estimate * (-estimate * values + np.float32(2.0))


def approx_low_inverse_sqrt(values: np.ndarray) -> np.ndarray:
    """The reference's APrxLoRsqF1 integer-magic inverse square root."""
    return (LOW_INVERSE_SQRT_BITS - (values.view(np.uint32) >> np.uint32(1))).view(np.float32)
