import inspect
import sys
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path

from PIL import Image, ImageChops
from tqdm.contrib.concurrent import process_map

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


def save_preview(image: Image.Image, path: Path) -> str:
    relative_path = path.relative_to(DOCS_DIRECTORY.parent)
    if not differs_from_existing(image, path):
        return f"skipped {relative_path} (unchanged)"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return f"wrote {relative_path}"


def preview_output_path(prefix: str, direction: str, sample_name: str) -> Path:
    extension = ".jpg" if direction == "upscale" and sample_name == "photo" else ".png"
    return PREVIEWS_DIRECTORY / f"{prefix}_{direction}_{sample_name}{extension}"


@dataclass(frozen=True)
class PreviewJob:
    algorithm: str
    task: PreviewTask


def build_jobs() -> list[PreviewJob]:
    jobs: list[PreviewJob] = []
    for name in scalerack.ALGORITHMS:
        has_factor = accepts_factor(name)
        for task in PREVIEW_TASKS:
            if not has_factor and task.factor < 1:
                continue
            jobs.append(PreviewJob(name, task))
    return jobs


def render_preview(job: PreviewJob) -> str | None:
    has_factor = accepts_factor(job.algorithm)
    source = prepare_input(job.task)
    # noinspection PyBroadException
    try:
        result = (
            scalerack.resize(job.algorithm, source, job.task.factor)
            if has_factor
            else scalerack.resize(job.algorithm, source)
        )
    except Exception:
        return None
    direction = classify_resize(source, result)
    if direction is None:
        return None
    return save_preview(
        result, PREVIEWS_DIRECTORY / f"{job.algorithm}_{direction}_{job.task.name}.png"
    )


def generate_algorithm_previews() -> None:
    jobs = build_jobs()
    messages = process_map(
        render_preview, jobs, desc="rendering previews", chunksize=1, max_workers=cpu_count()
    )
    for message in messages:
        if message is not None:
            print(message)


def main() -> int:
    generate_algorithm_previews()
    print(f"previews complete for {len(scalerack.ALGORITHMS)} algorithms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
