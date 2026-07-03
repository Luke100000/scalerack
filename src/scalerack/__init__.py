import inspect
from importlib import metadata
from typing import Any, cast

from scalerack.algorithms.bicubic import bicubic
from scalerack.algorithms.bilinear import bilinear
from scalerack.algorithms.box import box
from scalerack.algorithms.catmull_rom import catmull_rom
from scalerack.algorithms.content_adaptive_downscale import content_adaptive_downscale
from scalerack.algorithms.epx import scale2x, scale3x, scale4x
from scalerack.algorithms.lanczos import lanczos
from scalerack.algorithms.magic_kernel_sharp import magic_kernel_sharp
from scalerack.algorithms.mitchell import mitchell
from scalerack.algorithms.nearest import nearest
from scalerack.algorithms.registry import ALGORITHMS
from scalerack.exceptions import (
    InvalidFactorError,
    ScalerackError,
    UnknownAlgorithmError,
    UnsupportedImageError,
)
from scalerack.image_io import ImageT, to_array
from scalerack.validation import derive_factor, resolve_output_size

__all__ = [
    "ALGORITHMS",
    "ImageT",
    "InvalidFactorError",
    "ScalerackError",
    "UnknownAlgorithmError",
    "UnsupportedImageError",
    "__version__",
    "bicubic",
    "bilinear",
    "box",
    "catmull_rom",
    "content_adaptive_downscale",
    "lanczos",
    "magic_kernel_sharp",
    "mitchell",
    "nearest",
    "resize",
    "scale2x",
    "scale3x",
    "scale4x",
]

try:
    __version__ = metadata.version("scalerack")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"


def resize(
    method: str,
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    **opts: Any,
) -> ImageT:
    """Scale an image with the algorithm named by ``method``.

    Translates the request into whatever sizing the algorithm exposes:
    a factor becomes width/height (and vice versa) where needed.

    Args:
        method: Algorithm name (a key of ``ALGORITHMS``).
        **opts: Algorithm-specific tuning options (e.g. ``taps=4`` for lanczos).
    """
    if method not in ALGORITHMS:
        raise UnknownAlgorithmError(
            f"unknown algorithm '{method}'; available algorithms: {sorted(ALGORITHMS)}"
        )

    function = ALGORITHMS[method]
    parameters = inspect.signature(function).parameters
    accepts_factor = "factor" in parameters
    accepts_dimensions = "width" in parameters and "height" in parameters

    if accepts_factor and accepts_dimensions:
        return cast(ImageT, function(image, factor=factor, width=width, height=height, **opts))

    if accepts_factor:
        if width is not None or height is not None:
            array = to_array(image)
            factor = derive_factor(array.shape[0], array.shape[1], width, height)
        if factor is None:
            return cast(ImageT, function(image, **opts))
        return cast(ImageT, function(image, factor=factor, **opts))

    if accepts_dimensions:
        if factor is not None:
            array = to_array(image)
            output_height, output_width = resolve_output_size(
                array.shape[0], array.shape[1], factor, width, height
            )
            width, height = output_width, output_height
        return cast(ImageT, function(image, width=width, height=height, **opts))

    return cast(ImageT, function(image, **opts))
