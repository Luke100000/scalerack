from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput
from scalerack.kernels import make_magic_kernel_sharp
from scalerack.resample import resample_with_kernel


@register
def magic_kernel_sharp(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    version: int = 2021,
) -> ImageInput:
    """Scale with Costella's Magic Kernel Sharp.

    A crisp modern alternative to plain cubic kernels.

    Args:
        version: 2013 (slightly sharpening, cheaper) or 2021 (spectrally flatter).
    """
    kernel = make_magic_kernel_sharp(version)
    return resample_with_kernel(image, kernel, factor=factor, width=width, height=height)
