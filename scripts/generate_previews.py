"""Regenerate the README preview gallery from the checked-in samples.

Run from the repository root: ``python scripts/generate_previews.py``
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import scalerack

DOCS_DIRECTORY = Path(__file__).resolve().parent.parent / "docs"
SAMPLES_DIRECTORY = DOCS_DIRECTORY / "samples"
PREVIEWS_DIRECTORY = DOCS_DIRECTORY / "previews"

DEFAULT_PREVIEW_FACTOR = 2
FIXED_PREVIEW_FACTORS = {"scale2x": 2, "scale3x": 3, "scale4x": 4}

LABEL_BAR_HEIGHT = 22
LABEL_MARGIN = 4
GUTTER = 6
BACKGROUND = (248, 248, 248, 255)
LABEL_COLOR = (20, 20, 20, 255)


def load_label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=13)
    except TypeError:  # older Pillow without the size parameter
        return ImageFont.load_default()


def compose_comparison(reference: Image.Image, result: Image.Image, title: str) -> Image.Image:
    """Side-by-side sheet: nearest-scaled original on the left, algorithm output right."""
    font = load_label_font()
    reference_label = "nearest (reference)"
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    label_width = 2 * LABEL_MARGIN + max(
        int(probe.textlength(reference_label, font)), int(probe.textlength(title, font))
    )
    column_width = max(reference.width, result.width, label_width)
    width = 2 * column_width + 3 * GUTTER
    height = max(reference.height, result.height) + LABEL_BAR_HEIGHT + 2 * GUTTER
    sheet = Image.new("RGBA", (width, height), BACKGROUND)
    left_column = GUTTER
    right_column = column_width + 2 * GUTTER
    sheet.paste(reference, (left_column, LABEL_BAR_HEIGHT + GUTTER), mask=reference)
    sheet.paste(result, (right_column, LABEL_BAR_HEIGHT + GUTTER), mask=result)
    draw = ImageDraw.Draw(sheet)
    draw.text((left_column + LABEL_MARGIN, LABEL_MARGIN), reference_label, LABEL_COLOR, font)
    draw.text((right_column + LABEL_MARGIN, LABEL_MARGIN), title, LABEL_COLOR, font)
    return sheet


def main() -> int:
    PREVIEWS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    samples = {
        path.stem: Image.open(path).convert("RGBA")
        for path in sorted(SAMPLES_DIRECTORY.glob("*.png"))
    }
    if not samples:
        print(f"no samples found in {SAMPLES_DIRECTORY}; run scripts/make_samples.py first")
        return 1

    for name in scalerack.ALGORITHMS:
        factor = FIXED_PREVIEW_FACTORS.get(name, DEFAULT_PREVIEW_FACTOR)
        for sample_name, source in samples.items():
            result = scalerack.resize(name, source, factor)
            reference = scalerack.nearest(source, factor)
            sheet = compose_comparison(reference, result, f"{name} ({factor}x)")
            output_path = PREVIEWS_DIRECTORY / f"{name}_{sample_name}.png"
            sheet.save(output_path)
            print(f"wrote {output_path.relative_to(DOCS_DIRECTORY.parent)}")

    print(f"previews complete for {len(scalerack.ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
