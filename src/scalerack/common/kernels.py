from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

WeightFunction = Callable[[np.ndarray], np.ndarray]

MAGIC_KERNEL_SHARP_TAPS = {
    2013: (-1 / 4, 3 / 2, -1 / 4),
    2021: tuple(tap / 144 for tap in (-1, 6, -35, 204, -35, 6, -1)),
}


@dataclass(frozen=True)
class Kernel:
    """A continuous resampling kernel: its weight function and support radius."""

    support: float
    evaluate: WeightFunction


def make_triangle_kernel() -> Kernel:
    """Triangle (tent) kernel — linear interpolation."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(x))

    return Kernel(support=1.0, evaluate=evaluate)


def make_keys_cubic_kernel(a: float) -> Kernel:
    """Keys (1981) cubic kernel; ``a`` trades sharpness against overshoot (classic bicubic)."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        ax = np.abs(x)
        inner = (a + 2) * ax**3 - (a + 3) * ax**2 + 1
        outer = a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a
        return np.where(ax <= 1, inner, np.where(ax < 2, outer, 0.0))

    return Kernel(support=2.0, evaluate=evaluate)


def make_bc_spline_kernel(b: float, c: float) -> Kernel:
    """Mitchell-Netravali (1988) BC-spline family (B=C=1/3 Mitchell; B=0, C=1/2 Catmull-Rom)."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        ax = np.abs(x)
        inner = ((12 - 9 * b - 6 * c) * ax**3 + (-18 + 12 * b + 6 * c) * ax**2 + (6 - 2 * b)) / 6
        outer = (
            (-b - 6 * c) * ax**3
            + (6 * b + 30 * c) * ax**2
            + (-12 * b - 48 * c) * ax
            + (8 * b + 24 * c)
        ) / 6
        return np.where(ax < 1, inner, np.where(ax < 2, outer, 0.0))

    return Kernel(support=2.0, evaluate=evaluate)


def make_lanczos_kernel(taps: int) -> Kernel:
    """Lanczos windowed-sinc kernel with the given number of lobes."""
    if not isinstance(taps, int) or taps < 1:
        raise ValueError(f"lanczos taps must be a positive integer, got {taps!r}")

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) < taps, np.sinc(x) * np.sinc(x / taps), 0.0)

    return Kernel(support=float(taps), evaluate=evaluate)


def evaluate_magic_kernel(x: np.ndarray) -> np.ndarray:
    """Costella's magic kernel m(x) — the quadratic B-spline with support 1.5.

    Definitions and sharpening taps from johncostella.com/magic.
    """
    ax = np.abs(x)
    near = 0.75 - ax**2
    far = 0.5 * (1.5 - ax) ** 2
    return np.where(ax <= 0.5, near, np.where(ax < 1.5, far, 0.0))


def make_magic_kernel_sharp(version: int) -> Kernel:
    """Magic Kernel Sharp: m(x) convolved with the version's discrete sharpening taps."""
    if version not in MAGIC_KERNEL_SHARP_TAPS:
        raise ValueError(
            f"magic kernel sharp version must be one of {sorted(MAGIC_KERNEL_SHARP_TAPS)}, "
            f"got {version!r}"
        )
    sharp_taps = MAGIC_KERNEL_SHARP_TAPS[version]
    radius = len(sharp_taps) // 2

    def evaluate(x: np.ndarray) -> np.ndarray:
        result = np.zeros(np.shape(x), dtype=np.float64)
        for offset, weight in enumerate(sharp_taps, start=-radius):
            result += weight * evaluate_magic_kernel(x - offset)
        return result

    return Kernel(support=1.5 + radius, evaluate=evaluate)
