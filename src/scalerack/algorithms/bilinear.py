from scalerack.algorithms.registry import register
from scalerack.common.kernels import make_triangle_kernel
from scalerack.common.resample import resample_with_kernel
from scalerack.image_io import ImageInput


@register
def bilinear(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Scale with linear interpolation over the triangle kernel.

    A fast, artifact-light baseline; visibly softer than the cubic and
    sinc families.
    """
    return resample_with_kernel(
        image, make_triangle_kernel(), factor=factor, width=width, height=height
    )
