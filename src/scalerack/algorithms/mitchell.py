from scalerack.algorithms.registry import register
from scalerack.common.kernels import make_bc_spline_kernel
from scalerack.common.resample import resample_with_kernel
from scalerack.image_io import ImageInput


@register
def mitchell(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    b: float = 1 / 3,
    c: float = 1 / 3,
) -> ImageInput:
    """Scale with the Mitchell-Netravali BC-spline.

    The safe offline default balancing blur, anisotropy, and ringing.

    Args:
        b: BC-spline B parameter.
        c: BC-spline C parameter.
    """
    return resample_with_kernel(
        image, make_bc_spline_kernel(b, c), factor=factor, width=width, height=height
    )
