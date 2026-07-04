from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput
from scalerack.kernels import make_bc_spline_kernel
from scalerack.resample import resample_with_kernel


@register
def catmull_rom(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> ImageInput:
    """Scale with the Catmull-Rom interpolating spline.

    Sharper than Mitchell; a good pick when extra acutance is worth a
    little more ringing.
    """
    kernel = make_bc_spline_kernel(b=0.0, c=0.5)
    return resample_with_kernel(image, kernel, factor=factor, width=width, height=height)
