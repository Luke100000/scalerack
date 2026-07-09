from scalerack.algorithms.registry import register
from scalerack.common.kernels import make_lanczos_kernel
from scalerack.common.resample import resample_with_kernel
from scalerack.image_io import ImageInput


@register
def lanczos(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    taps: int = 3,
) -> ImageInput:
    """Scale with a Lanczos windowed-sinc kernel.

    The high-detail standard for photographic content; may ring at hard edges.

    Args:
        taps: Number of sinc lobes; more is sharper, slower, and more ring-prone.
    """
    return resample_with_kernel(
        image, make_lanczos_kernel(taps), factor=factor, width=width, height=height
    )
