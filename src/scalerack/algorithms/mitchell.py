from scalerack.algorithms.registry import register
from scalerack.image_io import ImageT
from scalerack.kernels import make_bc_spline_kernel
from scalerack.resample import resample_with_kernel


@register
def mitchell(
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    b: float = 1 / 3,
    c: float = 1 / 3,
) -> ImageT:
    """Scale with the Mitchell-Netravali BC-spline.

    The safe offline default balancing blur, anisotropy, and ringing.

    Args:
        b: BC-spline B parameter.
        c: BC-spline C parameter.
    """
    return resample_with_kernel(
        image, make_bc_spline_kernel(b, c), factor=factor, width=width, height=height
    )
