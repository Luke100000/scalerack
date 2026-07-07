import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops

import scalerack

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
    reconstruct_from_downscale: bool = False


PREVIEW_TASKS = (
    PreviewTask("photo", SAMPLES_DIRECTORY / "photo_downscale.jpg", DOWNSCALE_FACTOR),
    PreviewTask("sprite", SAMPLES_DIRECTORY / "sprite_downscale.png", DOWNSCALE_FACTOR),
    PreviewTask(
        "photo",
        SAMPLES_DIRECTORY / "photo_upscale.jpg",
        UPSCALE_FACTOR,
        reconstruct_from_downscale=True,
    ),
    PreviewTask("sprite", SAMPLES_DIRECTORY / "sprite_upscale.png", UPSCALE_FACTOR),
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


MAX_UNCHANGED_CHANNEL_DIFFERENCE = 1


def differs_from_existing(image: Image.Image, path: Path) -> bool:
    if not path.exists():
        return True
    with Image.open(path) as existing:
        existing = existing.convert(image.mode)
        if existing.size != image.size:
            return True
        difference = ImageChops.difference(image, existing)
    max_channel_difference = max(band_maximum for _, band_maximum in difference.getextrema())
    return max_channel_difference > MAX_UNCHANGED_CHANNEL_DIFFERENCE


def save_preview(image: Image.Image, path: Path) -> None:
    if not differs_from_existing(image, path):
        print(f"skipped {path.relative_to(DOCS_DIRECTORY.parent)} (unchanged)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"wrote {path.relative_to(DOCS_DIRECTORY.parent)}")


def preview_output_path(prefix: str, direction: str, sample_name: str) -> Path:
    extension = ".jpg" if direction == "upscale" and sample_name == "photo" else ".png"
    return PREVIEWS_DIRECTORY / f"{prefix}_{direction}_{sample_name}{extension}"


def generate_algorithm_previews() -> None:
    for name in scalerack.ALGORITHMS:
        has_factor = accepts_factor(name)
        for task in PREVIEW_TASKS:
            if not has_factor and task.factor < 1:
                continue

            source = prepare_input(task)
            # noinspection PyBroadException
            try:
                result = (
                    scalerack.resize(name, source, task.factor)
                    if has_factor
                    else scalerack.resize(name, source)
                )
            except Exception:
                continue
            direction = classify_resize(source, result)
            if direction is None:
                continue
            save_preview(result, PREVIEWS_DIRECTORY / f"{name}_{direction}_{task.name}.png")


def main() -> int:
    generate_algorithm_previews()
    print(f"previews complete for {len(scalerack.ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
