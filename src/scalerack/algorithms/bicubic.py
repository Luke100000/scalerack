from scalerack.algorithms.registry import register
from scalerack.common.kernels import make_keys_cubic_kernel
from scalerack.common.resample import resample_with_kernel
from scalerack.image_io import ImageInput


@register()
def bicubic(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    a: float = -0.5,
) -> ImageInput:
    """Scale with the classic Keys bicubic kernel.

    Sharper than bilinear with mild overshoot; the general-purpose default
    when Lanczos rings too much.

    Args:
        a: Keys sharpness coefficient (-0.5 standard; -0.75 sharper, more halo-prone).
    """
    return resample_with_kernel(
        image, make_keys_cubic_kernel(a), factor=factor, width=width, height=height
    )
