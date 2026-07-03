import math
import numbers

from scalerack.exceptions import InvalidFactorError

MINIMUM_OUTPUT_DIMENSION = 1


def resolve_output_size(
    input_height: int,
    input_width: int,
    factor: float | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Return the validated output (height, width) for a sizing request."""
    has_dimensions = width is not None or height is not None
    if factor is not None and has_dimensions:
        raise InvalidFactorError("provide either 'factor' or 'width'/'height', not both")
    if factor is not None:
        return resolve_from_factor(input_height, input_width, factor)
    if has_dimensions:
        return resolve_from_dimensions(input_height, input_width, width, height)
    raise InvalidFactorError("provide either 'factor' or 'width'/'height'")


def resolve_from_factor(input_height: int, input_width: int, factor: float) -> tuple[int, int]:
    """Validate a factor request and compute output dimensions as round(dim * factor)."""
    if isinstance(factor, bool) or not isinstance(factor, numbers.Real):
        raise InvalidFactorError(f"factor must be a positive number, got {factor!r}")
    value = float(factor)
    if not math.isfinite(value) or value <= 0:
        raise InvalidFactorError(f"factor must be a positive finite number, got {factor!r}")
    output_height = max(MINIMUM_OUTPUT_DIMENSION, round(input_height * value))
    output_width = max(MINIMUM_OUTPUT_DIMENSION, round(input_width * value))
    return output_height, output_width


def resolve_from_dimensions(
    input_height: int, input_width: int, width: int | None, height: int | None
) -> tuple[int, int]:
    """Complete a partial width/height request by preserving the aspect ratio."""
    validate_dimension("width", width)
    validate_dimension("height", height)
    if width is None:
        assert height is not None
        width = max(MINIMUM_OUTPUT_DIMENSION, round(height * input_width / input_height))
    if height is None:
        height = max(MINIMUM_OUTPUT_DIMENSION, round(width * input_height / input_width))
    return height, width


def derive_factor(
    input_height: int, input_width: int, width: int | None, height: int | None
) -> float:
    """Translate a dimension request into one factor, for factor-only algorithms."""
    output_height, output_width = resolve_from_dimensions(input_height, input_width, width, height)
    factor = output_width / input_width
    if max(MINIMUM_OUTPUT_DIMENSION, round(input_height * factor)) != output_height:
        raise InvalidFactorError(
            f"no single factor maps {input_width}x{input_height} to {output_width}x{output_height}"
        )
    return factor


def validate_dimension(name: str, value: int | None) -> None:
    """Reject non-positive or non-integer output dimensions."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidFactorError(f"{name} must be a positive integer, got {value!r}")
