import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import scalerack

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIRECTORY = ROOT / "docs"
OUTPUT_PATH = DOCS_DIRECTORY / "index.md"
CONFIG_PATH = DOCS_DIRECTORY / "_config.yml"
GEMFILE_PATH = DOCS_DIRECTORY / "Gemfile"
REPOSITORY_URL = "https://github.com/Luke100000/scalerack"
PARAMETER_PATTERN = re.compile(r"^\s{4}([^:\s]+):\s*(.*)$")


@dataclass(frozen=True)
class ParsedDocstring:
    summary: str
    description: str
    parameters: dict[str, str]


def parse_docstring(obj: object) -> ParsedDocstring:
    text_lines: list[str] = []
    parameter_lines: list[str] = []
    in_args = False

    for line in (inspect.getdoc(obj) or "").splitlines():
        if line.strip() == "Args:":
            in_args = True
        elif in_args:
            parameter_lines.append(line)
        else:
            text_lines.append(line)

    paragraphs = [
        " ".join(part.strip() for part in paragraph.splitlines()).strip()
        for paragraph in "\n".join(text_lines).split("\n\n")
        if paragraph.strip()
    ]
    return ParsedDocstring(
        summary=paragraphs[0] if paragraphs else "",
        description="\n".join(paragraphs[1:]),
        parameters=parse_parameters(parameter_lines),
    )


def parse_parameters(lines: list[str]) -> dict[str, str]:
    parameters: dict[str, str] = {}
    name: str | None = None
    description_parts: list[str] = []

    def save_parameter() -> None:
        if name is not None:
            parameters[name] = " ".join(description_parts).strip()

    for line in lines:
        if not line.strip():
            continue
        match = PARAMETER_PATTERN.match(line)
        if match is not None:
            save_parameter()
            name = match.group(1)
            description_parts = [match.group(2)]
        elif name is not None:
            description_parts.append(line.strip())

    save_parameter()
    return parameters


def source_url(obj: object) -> str:
    source_file = inspect.getsourcefile(obj)
    if source_file is None:
        return REPOSITORY_URL
    try:
        relative_path = Path(source_file).relative_to(ROOT)
    except ValueError:
        return REPOSITORY_URL
    _, line = inspect.getsourcelines(obj)
    return f"{REPOSITORY_URL}/blob/master/{relative_path.as_posix()}#L{line}"


def format_annotation(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    if isinstance(annotation, type):
        return annotation.__name__
    if hasattr(annotation, "__name__"):
        return str(annotation.__name__)
    return str(annotation).removeprefix("~").replace("scalerack.image_io.", "")


def parameter_description(name: str, parameter: inspect.Parameter, doc: ParsedDocstring) -> str:
    if name in doc.parameters:
        return doc.parameters[name]
    if name == "factor":
        return "Scale factor."
    if name == "width":
        return "Target output width."
    if name == "height":
        return "Target output height."
    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
        return "Algorithm-specific options."
    return ""


def render_parameters(function: object, doc: ParsedDocstring) -> str:
    parameters = inspect.signature(function).parameters.values()
    if not parameters:
        return ""

    lines = [""]
    for parameter in parameters:
        if parameter.name == "image":
            continue
        prefix = "**" if parameter.kind is inspect.Parameter.VAR_KEYWORD else ""
        annotation = format_annotation(parameter.annotation)
        details = []
        if annotation:
            details.append(f"`{annotation}`")
        if parameter.default is not inspect.Signature.empty:
            details.append(f"default `{parameter.default!r}`")
        declaration = f"`{prefix}{parameter.name}`"
        detail_text = f" ({', '.join(details)})" if details else ""
        description = parameter_description(parameter.name, parameter, doc)
        lines.append(f"- {declaration}{detail_text}: {description}".rstrip())
    return "\n".join(lines) if len(lines) > 2 else ""


def render_callable(name: str, function: object) -> str:
    doc = parse_docstring(function)
    parts = [f"### [`{name}`]({source_url(function)})", "", doc.summary or "No docstring yet."]
    if doc.description:
        parts.extend(["", doc.description])
    parts.extend(["", render_parameters(function, doc)])
    parts.extend(["", "---", "", "<br />"])
    return "\n".join(parts)


def render_algorithms() -> str:
    return "\n\n".join(
        render_callable(name, scalerack.ALGORITHMS[name].function)
        for name in sorted(scalerack.ALGORITHMS)
    )


def render_exceptions() -> str:
    rows = []
    for name in (
        "ScalerackError",
        "UnknownAlgorithmError",
        "InvalidFactorError",
        "UnsupportedImageError",
    ):
        exception = getattr(scalerack, name)
        summary = parse_docstring(exception).summary or "No docstring yet."
        rows.append(f"- [`{name}`]({source_url(exception)}) - {summary}")
    return "\n".join(rows)


def render_markdown() -> str:
    return f"""---
title: Scalerack Python API
layout: default
---

# Scalerack

Python image resizing, resampling, downscaling, and pixel-art upscaling for NumPy arrays
and Pillow images.

## Install

~~~bash
pip install scalerack
pip install scalerack[cli]
pip install scalerack[all]
~~~

## Usage

~~~python
import scalerack

big = scalerack.mitchell(image, factor=2.5)
thumb = scalerack.box(photo, width=320)
sprite = scalerack.scale3x(pil_sprite)
out = scalerack.resize("lanczos", image, factor=2)
~~~

## Algorithms

{render_algorithms()}

## API

{render_callable("resize", scalerack.resize)}

## Exceptions

{render_exceptions()}
"""


def write_project_files() -> None:
    CONFIG_PATH.write_text(
        """title: Scalerack
description: Python image resizing, resampling, downscaling, and pixel-art upscaling API reference.
theme: minima
markdown: kramdown
highlighter: rouge
kramdown:
  input: GFM
  syntax_highlighter: rouge
exclude:
  - _site
""",
        encoding="utf-8",
    )
    GEMFILE_PATH.write_text(
        """source "https://rubygems.org"

gem "jekyll", "~> 4.4"
gem "minima"
""",
        encoding="utf-8",
    )


def main() -> int:
    DOCS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_project_files()
    OUTPUT_PATH.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
