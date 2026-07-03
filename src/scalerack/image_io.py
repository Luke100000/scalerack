import sys
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from scalerack.constants import SUPPORTED_PIL_MODES
from scalerack.exceptions import UnsupportedImageError

if TYPE_CHECKING:
    from PIL import Image

ImageT = TypeVar("ImageT")

SUPPORTED_ARRAY_DTYPES = frozenset({"uint8", "float32", "float64"})
SUPPORTED_CHANNEL_COUNTS = frozenset({3, 4})
UINT8_MAX = 255


def is_pil_image(image: object) -> bool:
    """Detect a PIL image without importing Pillow.

    If the caller passes a PIL image, Pillow is necessarily already imported,
    so checking ``sys.modules`` keeps Pillow a soft dependency.
    """
    pil_image_module = sys.modules.get("PIL.Image")
    if pil_image_module is None:
        return False
    return isinstance(image, pil_image_module.Image)


def to_array(image: object) -> np.ndarray:
    """Convert a supported image representation into a validated contiguous array."""
    if isinstance(image, np.ndarray):
        return validate_array(image)
    if is_pil_image(image):
        pil_image = cast("Image.Image", image)
        if pil_image.mode not in SUPPORTED_PIL_MODES:
            raise UnsupportedImageError(
                f"unsupported PIL mode '{pil_image.mode}'; "
                f"supported modes: {sorted(SUPPORTED_PIL_MODES)}"
            )
        return np.asarray(pil_image)
    raise UnsupportedImageError(
        f"unsupported image type '{type(image).__name__}'; "
        "supported representations: numpy.ndarray and PIL.Image.Image"
    )


def validate_array(image: np.ndarray) -> np.ndarray:
    """Check dtype and shape against the supported contract; normalize contiguity."""
    if image.dtype.name not in SUPPORTED_ARRAY_DTYPES:
        raise UnsupportedImageError(
            f"unsupported array dtype '{image.dtype.name}'; "
            f"supported dtypes: {sorted(SUPPORTED_ARRAY_DTYPES)}"
        )
    is_grayscale = image.ndim == 2
    is_multichannel = image.ndim == 3 and image.shape[2] in SUPPORTED_CHANNEL_COUNTS
    if not (is_grayscale or is_multichannel):
        raise UnsupportedImageError(
            f"unsupported array shape {image.shape}; "
            "supported shapes: (H, W) grayscale, (H, W, 3) RGB, (H, W, 4) RGBA"
        )
    if image.size == 0:
        raise UnsupportedImageError("input image has zero pixels")
    return np.ascontiguousarray(image)


def from_array(result: np.ndarray, original: object) -> Any:
    """Return the result in the same representation as the original input."""
    if is_pil_image(original):
        from PIL import Image as pil_image_module

        return pil_image_module.fromarray(result)
    return result


def restore_dtype(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Cast float computation results back to the input dtype (clip and round for uint8)."""
    if dtype == np.uint8:
        return np.clip(np.rint(values), 0, UINT8_MAX).astype(np.uint8)
    return values.astype(dtype, copy=False)
