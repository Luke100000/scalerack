from scalerack.algorithms.registry import register
from scalerack.image_io import ImageT
from scalerack.kernels import make_triangle_kernel
from scalerack.resample import resample_with_kernel


@register
def bilinear(
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageT:
    """Scale with linear interpolation over the triangle kernel.

    A fast, artifact-light baseline; visibly softer than the cubic and
    sinc families.
    """
    return resample_with_kernel(
        image, make_triangle_kernel(), factor=factor, width=width, height=height
    )
