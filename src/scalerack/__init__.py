import inspect
from importlib import metadata
from typing import TypeVar

from scalerack.algorithms.bicubic import bicubic
from scalerack.algorithms.bilinear import bilinear
from scalerack.algorithms.box import box
from scalerack.algorithms.catmull_rom import catmull_rom
from scalerack.algorithms.channel_median import channel_median
from scalerack.algorithms.content_adaptive_downscale import content_adaptive_downscale
from scalerack.algorithms.depixelize import depixelize
from scalerack.algorithms.eagle import eagle2x, eagle3x
from scalerack.algorithms.epx import scale2x, scale3x, scale4x
from scalerack.algorithms.hqx import hq2x, hq3x, hq4x
from scalerack.algorithms.lanczos import lanczos
from scalerack.algorithms.magic_kernel_sharp import magic_kernel_sharp
from scalerack.algorithms.mitchell import mitchell
from scalerack.algorithms.nearest import nearest
from scalerack.algorithms.registry import ALGORITHMS, Algorithm
from scalerack.algorithms.sai import sai2x, super2xsai, supereagle
from scalerack.algorithms.superxbr import superxbr
from scalerack.algorithms.vector_median import vector_median
from scalerack.algorithms.xbrz import xbrz
from scalerack.exceptions import (
    InvalidFactorError,
    ScalerackError,
    UnknownAlgorithmError,
    UnsupportedImageError,
)
from scalerack.image_io import ImageInput, as_image_input
from scalerack.validation import derive_factor, resolve_output_size

__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "ImageInput",
    "InvalidFactorError",
    "ScalerackError",
    "UnknownAlgorithmError",
    "UnsupportedImageError",
    "__version__",
    "bicubic",
    "bilinear",
    "box",
    "catmull_rom",
    "channel_median",
    "content_adaptive_downscale",
    "depixelize",
    "eagle2x",
    "eagle3x",
    "hq2x",
    "hq3x",
    "hq4x",
    "lanczos",
    "magic_kernel_sharp",
    "mitchell",
    "nearest",
    "resize",
    "sai2x",
    "scale2x",
    "scale3x",
    "scale4x",
    "super2xsai",
    "supereagle",
    "superxbr",
    "vector_median",
    "xbrz",
]

try:
    __version__ = metadata.version("scalerack")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"


ImageT = TypeVar("ImageT")


def resize(
    method: str,
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    **opts: object,
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

    function = ALGORITHMS[method].function
    parameters = inspect.signature(function).parameters
    accepts_factor = "factor" in parameters
    accepts_dimensions = "width" in parameters and "height" in parameters

    if accepts_factor and accepts_dimensions:
        return function(image, factor=factor, width=width, height=height, **opts)

    if accepts_factor:
        if width is not None or height is not None:
            array = as_image_input(image).numpy()
            factor = derive_factor(array.shape[0], array.shape[1], width, height)
        if factor is None:
            return function(image, **opts)
        return function(image, factor=factor, **opts)

    if accepts_dimensions:
        if factor is not None:
            array = as_image_input(image).numpy()
            output_height, output_width = resolve_output_size(
                array.shape[0], array.shape[1], factor, width, height
            )
            width, height = output_width, output_height
        return function(image, width=width, height=height, **opts)

    return function(image, **opts)
