import numpy as np
import pytest

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96
RANDOM_SEED = 20260703
NOISE_MAX = 16
CHECKER_CELL = 4
CHECKER_BRIGHTNESS = 60


@pytest.fixture
def complex_image() -> np.ndarray:
    """A complex RGBA image: gradients, hard edges, a checkerboard, an alpha ramp, and noise."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = np.linspace(0.0, 1.0, IMAGE_HEIGHT)[:, None]
    cols = np.linspace(0.0, 1.0, IMAGE_WIDTH)[None, :]
    red = 255.0 * rows * np.ones_like(cols)
    green = 255.0 * cols * np.ones_like(rows)
    blue = np.where((rows > 0.5) ^ (cols > 0.5), 255.0, 0.0)
    checker_rows = np.arange(IMAGE_HEIGHT)[:, None] // CHECKER_CELL
    checker_cols = np.arange(IMAGE_WIDTH)[None, :] // CHECKER_CELL
    checker = ((checker_rows + checker_cols) % 2) * CHECKER_BRIGHTNESS
    alpha = 255.0 * (0.25 + 0.75 * cols) * np.ones_like(rows)
    stacked = np.stack([red + checker, green, blue, alpha], axis=-1)
    noise = rng.integers(0, NOISE_MAX, stacked.shape)
    return np.clip(stacked + noise, 0, 255).astype(np.uint8)
