from pathlib import Path

import cyclopts
from PIL import Image

import scalerack
from scalerack.constants import SUPPORTED_PIL_MODES

app = cyclopts.App(
    name="scalerack",
    version=scalerack.__version__,
    result_action="return_value",
)

OptionValue = int | float | str


@app.command
def scale(
    input_path: Path,
    output_path: Path,
    *,
    factor: float | None = None,
    width: int | None = None,
    height: int | None = None,
    method: str = "lanczos",
    opt: list[str] | None = None,
) -> None:
    """Scale an image file.

    Args:
        input_path: Readable input image file.
        output_path: Output path; format inferred from the extension.
        factor: Scale multiplier; mutually exclusive with width/height.
        width: Output width; height is inferred from the aspect ratio if omitted.
        height: Output height; width is inferred from the aspect ratio if omitted.
        method: Algorithm name (see ``scalerack list``).
        opt: Algorithm-specific KEY=VALUE option, repeatable (e.g. --opt taps=4).
    """
    with Image.open(input_path) as source:
        normalized = normalize_mode(source)
        options = parse_options(opt or [])
        result = scalerack.resize(method, normalized, factor, width=width, height=height, **options)
        result.save(output_path)
        print(
            f"{output_path} ({normalized.width}x{normalized.height} -> "
            f"{result.width}x{result.height}, {method})"
        )


@app.command(name="list")
def list_algorithms() -> None:
    """List every algorithm with its one-line summary."""
    for name, function in sorted(scalerack.ALGORITHMS.items()):
        summary = (function.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<20} {summary}")


def parse_options(pairs: list[str]) -> dict[str, OptionValue]:
    """Parse repeated ``KEY=VALUE`` options, coercing values to int or float when possible."""
    options: dict[str, OptionValue] = {}
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE for --opt, got {pair!r}")
        options[key] = coerce_option_value(raw_value)
    return options


def coerce_option_value(raw_value: str) -> OptionValue:
    """Interpret an option value as int, then float, then plain string."""
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def normalize_mode(image: Image.Image) -> Image.Image:
    """Convert unsupported PIL modes to RGBA (alpha-bearing or palette) or RGB."""
    if image.mode in SUPPORTED_PIL_MODES:
        return image
    has_alpha = "A" in image.mode or image.mode == "P"
    return image.convert("RGBA" if has_alpha else "RGB")
