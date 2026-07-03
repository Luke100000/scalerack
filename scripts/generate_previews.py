"""Regenerate the README preview gallery from the checked-in samples.

Run from the repository root: ``python scripts/generate_previews.py``
"""

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import scalerack
from scalerack.exceptions import InvalidFactorError

DOCS_DIRECTORY = Path(__file__).resolve().parent.parent / "docs"
SAMPLES_DIRECTORY = DOCS_DIRECTORY / "samples"
PREVIEWS_DIRECTORY = DOCS_DIRECTORY / "previews"

DOWNSCALE_FACTOR = 0.25
UPSCALE_FACTOR = 4
RECONSTRUCTION_DOWNSCALER = "lanczos"


@dataclass(frozen=True)
class PreviewTask:
    name: str
    path: Path
    factor: float
    original_direction: str
    reconstruct_from_downscale: bool = False


PREVIEW_TASKS = (
    PreviewTask("photo", SAMPLES_DIRECTORY / "photo_downscale.jpg", DOWNSCALE_FACTOR, "downscale"),
    PreviewTask(
        "sprite", SAMPLES_DIRECTORY / "sprite_downscale.png", DOWNSCALE_FACTOR, "downscale"
    ),
    PreviewTask(
        "photo",
        SAMPLES_DIRECTORY / "photo_upscale.jpg",
        UPSCALE_FACTOR,
        "upscale",
        reconstruct_from_downscale=True,
    ),
    PreviewTask("sprite", SAMPLES_DIRECTORY / "sprite_upscale.png", UPSCALE_FACTOR, "upscale"),
)


def accepts_factor(name: str) -> bool:
    parameters = inspect.signature(scalerack.ALGORITHMS[name]).parameters
    return "factor" in parameters


def open_sample(task: PreviewTask) -> Image.Image:
    with Image.open(task.path) as image:
        return image.convert("RGBA")


def prepare_input(task: PreviewTask) -> Image.Image:
    source = open_sample(task)
    if task.reconstruct_from_downscale:
        return scalerack.resize(RECONSTRUCTION_DOWNSCALER, source, DOWNSCALE_FACTOR)
    return source


def classify_resize(source: Image.Image, result: Image.Image) -> str | None:
    source_pixels = source.width * source.height
    result_pixels = result.width * result.height
    if result_pixels > source_pixels:
        return "upscale"
    if result_pixels < source_pixels:
        return "downscale"
    return None


def save_preview(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"wrote {path.relative_to(DOCS_DIRECTORY.parent)}")


def generate_originals() -> None:
    for task in PREVIEW_TASKS:
        source = open_sample(task)
        output_path = PREVIEWS_DIRECTORY / f"original_{task.original_direction}_{task.name}.png"
        save_preview(source, output_path)


def generate_algorithm_previews() -> None:
    for name in scalerack.ALGORITHMS:
        has_factor = accepts_factor(name)
        for task in PREVIEW_TASKS:
            if not has_factor and task.factor < 1:
                continue

            source = prepare_input(task)
            try:
                result = (
                    scalerack.resize(name, source, task.factor)
                    if has_factor
                    else scalerack.resize(name, source)
                )
            except InvalidFactorError:
                continue
            direction = classify_resize(source, result)
            if direction is None:
                continue
            save_preview(result, PREVIEWS_DIRECTORY / f"{name}_{direction}_{task.name}.png")


def main() -> int:
    missing_samples = [task.path for task in PREVIEW_TASKS if not task.path.exists()]
    if missing_samples:
        print("missing preview samples:")
        for path in missing_samples:
            print(f"  {path.relative_to(DOCS_DIRECTORY.parent)}")
        return 1

    generate_originals()
    generate_algorithm_previews()

    print(f"previews complete for {len(scalerack.ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
