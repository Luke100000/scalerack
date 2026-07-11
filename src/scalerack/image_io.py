import sys
from typing import TYPE_CHECKING, cast

import numpy as np

from scalerack.constants import SUPPORTED_PIL_MODES
from scalerack.exceptions import UnsupportedImageError
from scalerack.validation import resolve_output_size

if TYPE_CHECKING:
    from PIL import Image

SUPPORTED_ARRAY_DTYPES = frozenset({"uint8", "float32", "float64"})
SUPPORTED_CHANNEL_COUNTS = frozenset({3, 4})


def is_pil_image(image: object) -> bool:
    """Detect a PIL image without importing Pillow."""
    pil_image_module = sys.modules.get("PIL.Image")
    if pil_image_module is None:
        return False
    return isinstance(image, pil_image_module.Image)


def match_original_shape_and_dtype(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(result)
    if values.ndim == 3 and values.shape[2] == 4 and original.ndim == 2:
        values = values[:, :, :3].mean(axis=2)
    elif values.ndim == 3 and values.shape[2] == 4 and original.ndim == 3:
        channels = original.shape[2]
        if channels == 3:
            values = values[:, :, :3]
        elif channels == 4:
            values = values[:, :, :4]
    elif values.ndim == 3 and values.shape[2] == 3 and original.ndim == 3:
        channels = original.shape[2]
        if channels == 4:
            alpha = np.ones(values.shape[:2], dtype=values.dtype)
            if np.issubdtype(original.dtype, np.integer):
                alpha *= np.iinfo(original.dtype).max
            values = np.dstack((values, alpha))

    if (
        np.issubdtype(original.dtype, np.integer)
        and np.issubdtype(values.dtype, np.floating)
        and values.max(initial=0.0) <= 1.0
        and values.min(initial=0.0) >= 0.0
    ):
        values = values * np.iinfo(original.dtype).max
    return restore_dtype(values, original.dtype)


class ImageInput:
    """Wrap a raw image and expose validated raw and RGBA working views."""

    def __init__(self, image: object) -> None:
        self.raw = image

    @property
    def is_pil(self) -> bool:
        return is_pil_image(self.raw)

    def numpy(self) -> np.ndarray:
        return to_array(self.raw)

    def rgba(self) -> np.ndarray:
        """Return normalized ``float32`` RGBA pixels in the range ``[0, 1]``."""
        array = self.numpy()
        values = array.astype(np.float32, copy=False)
        if np.issubdtype(array.dtype, np.integer):
            values = (values / np.iinfo(array.dtype).max).astype(np.float32, copy=False)
        elif values.max(initial=0.0) > 1.0:
            values = (values / 255.0).astype(np.float32, copy=False)

        if values.ndim == 2:
            rgb = np.repeat(values[:, :, None], 3, axis=2)
            alpha = np.ones(values.shape, dtype=np.float32)
        elif values.shape[2] == 3:
            rgb = values
            alpha = np.ones(values.shape[:2], dtype=np.float32)
        else:
            rgb = values[:, :, :3]
            alpha = values[:, :, 3]

        return np.ascontiguousarray(np.dstack((rgb, alpha)).astype(np.float32, copy=False))

    def from_numpy(self, result: np.ndarray) -> "ImageInput":
        """Update the image, matching the original representation and dtype."""
        original = self.numpy()
        array = np.asarray(result)
        if array.size == 0:
            raise UnsupportedImageError("result image has zero pixels")

        restored = match_original_shape_and_dtype(array, original)
        self.raw = from_array(restored, self.raw)
        return self

    def get_target_dimensions(
        self,
        width: int | None = None,
        height: int | None = None,
        factor: float | None = None,
    ) -> tuple[int, int]:
        """Return validated target dimensions as ``(width, height)``."""
        input_height, input_width = self.numpy().shape[:2]
        output_height, output_width = resolve_output_size(
            input_height, input_width, factor, width, height
        )
        return output_width, output_height


def as_image_input(image: object) -> ImageInput:
    return image if isinstance(image, ImageInput) else ImageInput(image)


def to_array(image: object) -> np.ndarray:
    """Convert a supported image representation into a validated contiguous array."""
    if isinstance(image, ImageInput):
        return image.numpy()
    if isinstance(image, np.ndarray):
        return validate_array(image)
    if is_pil_image(image):
        pil_image = cast("Image.Image", image)
        if pil_image.mode in SUPPORTED_PIL_MODES:
            return np.asarray(pil_image)
        has_alpha = "A" in pil_image.mode or pil_image.mode == "P"
        pil_image = pil_image.convert("RGBA" if has_alpha else "RGB")
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


def from_array(result: np.ndarray, original: object) -> object:
    if is_pil_image(original):
        from PIL import Image as Image

        return Image.fromarray(result)
    return result


def restore_dtype(values: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Cast float computation results back to the input dtype (clip and round for uint8)."""
    if dtype == np.uint8:
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return values.astype(dtype, copy=False)
