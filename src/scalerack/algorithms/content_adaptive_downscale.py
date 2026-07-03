from dataclasses import dataclass
from typing import cast

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.exceptions import InvalidFactorError, UnsupportedImageError
from scalerack.image_io import (
    ImageT,
    from_array,
    remove_alpha_channel,
    restore_dtype,
    to_array,
)
from scalerack.resample import drop_channel_axis, ensure_channel_axis
from scalerack.validation import resolve_output_size

DEFAULT_ITERATIONS = 6
MIN_COLOR_VARIANCE = 1.0e-4
COLOR_VARIANCE_GROWTH = 1.1
EPSILON = 1.0e-12


@dataclass
class AdaptiveSample:
    center: np.ndarray
    covariance: np.ndarray
    color: np.ndarray
    color_variance: float


@register
def content_adaptive_downscale(
    image: ImageT,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> ImageT:
    """Downscale with content-adaptive kernels that preserve edge detail."""
    array = to_array(image)
    visible = remove_alpha_channel(array)
    validate_finite_values(visible)
    output_height, output_width = resolve_output_size(
        visible.shape[0], visible.shape[1], factor, width, height
    )
    validate_downscale(visible.shape[0], visible.shape[1], output_height, output_width)
    validate_iterations(iterations)

    compute_dtype = np.float32 if visible.dtype == np.uint8 else visible.dtype
    values = ensure_channel_axis(visible).astype(compute_dtype, copy=False)
    filtered = adaptive_downscale(values, output_height, output_width, iterations)
    restored = restore_dtype(filtered, visible.dtype)
    result = drop_channel_axis(restored, visible.ndim)
    return cast(ImageT, from_array(result, image))


def validate_finite_values(array: np.ndarray) -> None:
    """Reject NaN and infinite float values before iterative processing."""
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise UnsupportedImageError("content_adaptive_downscale requires finite pixel values")


def validate_downscale(
    input_height: int, input_width: int, output_height: int, output_width: int
) -> None:
    """Content-adaptive downscaling is defined only for true downscales."""
    if output_height >= input_height or output_width >= input_width:
        raise InvalidFactorError(
            "content_adaptive_downscale requires output width and height "
            "to be smaller than the input"
        )


def validate_iterations(iterations: int) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise InvalidFactorError(f"iterations must be a positive integer, got {iterations!r}")


def adaptive_downscale(
    values: np.ndarray, output_height: int, output_width: int, iterations: int
) -> np.ndarray:
    """Optimize one adaptive sample per output pixel using local EM-style updates."""
    input_height, input_width, channels = values.shape
    ratio_x = input_width / output_width
    ratio_y = input_height / output_height
    nominal_centers = make_nominal_centers(output_height, output_width, ratio_x, ratio_y)
    samples = initialize_samples(values, nominal_centers, ratio_x, ratio_y)
    pixel_positions = make_pixel_positions(input_height, input_width)
    regions = make_support_regions(input_height, input_width, nominal_centers, ratio_x, ratio_y)

    responsibilities: list[np.ndarray] = []
    for _ in range(iterations):
        responsibilities = expectation_step(values, pixel_positions, samples, regions)
        previous = sample_state(samples)
        maximization_step(values, pixel_positions, samples, regions, responsibilities)
        correction_step(samples, nominal_centers, output_height, output_width, ratio_x, ratio_y)
        if has_converged(previous, sample_state(samples)):
            break

    output = np.array([sample.color for sample in samples], dtype=values.dtype)
    return output.reshape(output_height, output_width, channels)


def make_nominal_centers(
    output_height: int, output_width: int, ratio_x: float, ratio_y: float
) -> np.ndarray:
    cols = (np.arange(output_width, dtype=np.float64) + 0.5) * ratio_x - 0.5
    rows = (np.arange(output_height, dtype=np.float64) + 0.5) * ratio_y - 0.5
    grid_x, grid_y = np.meshgrid(cols, rows)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def initialize_samples(
    values: np.ndarray, nominal_centers: np.ndarray, ratio_x: float, ratio_y: float
) -> list[AdaptiveSample]:
    covariance = np.array(
        [[max((ratio_x / 3.0) ** 2, 0.05), 0.0], [0.0, max((ratio_y / 3.0) ** 2, 0.05)]],
        dtype=np.float64,
    )
    neutral = np.full(values.shape[2], 0.5, dtype=np.float64)
    return [
        AdaptiveSample(
            center=center.copy(),
            covariance=covariance.copy(),
            color=neutral.copy(),
            color_variance=MIN_COLOR_VARIANCE,
        )
        for center in nominal_centers
    ]


def make_pixel_positions(input_height: int, input_width: int) -> np.ndarray:
    rows, cols = np.indices((input_height, input_width), dtype=np.float64)
    return np.stack([cols.ravel(), rows.ravel()], axis=1)


def make_support_regions(
    input_height: int,
    input_width: int,
    nominal_centers: np.ndarray,
    ratio_x: float,
    ratio_y: float,
) -> list[np.ndarray]:
    regions: list[np.ndarray] = []
    for center_x, center_y in nominal_centers:
        left = max(0, int(np.floor(center_x - 2.0 * ratio_x)))
        right = min(input_width - 1, int(np.ceil(center_x + 2.0 * ratio_x)))
        top = max(0, int(np.floor(center_y - 2.0 * ratio_y)))
        bottom = min(input_height - 1, int(np.ceil(center_y + 2.0 * ratio_y)))
        xs = np.arange(left, right + 1, dtype=np.int64)
        ys = np.arange(top, bottom + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        regions.append((grid_y.ravel() * input_width + grid_x.ravel()).astype(np.int64))
    return regions


def expectation_step(
    values: np.ndarray,
    pixel_positions: np.ndarray,
    samples: list[AdaptiveSample],
    regions: list[np.ndarray],
) -> list[np.ndarray]:
    flattened = values.reshape(-1, values.shape[2])
    per_sample_weights: list[np.ndarray] = []
    pixel_totals = np.zeros(flattened.shape[0], dtype=np.float64)

    for sample, region in zip(samples, regions, strict=True):
        positions = pixel_positions[region]
        colors = flattened[region].astype(np.float64, copy=False)
        spatial_weights = gaussian_spatial_weights(positions, sample)
        color_weights = gaussian_color_weights(colors, sample)
        weights = spatial_weights * color_weights
        total = weights.sum()
        if total <= EPSILON:
            weights = np.full(region.shape, 1.0 / region.size, dtype=np.float64)
        else:
            weights = weights / total
        per_sample_weights.append(weights)
        pixel_totals[region] += weights

    responsibilities: list[np.ndarray] = []
    for region, weights in zip(regions, per_sample_weights, strict=True):
        totals = pixel_totals[region]
        responsibilities.append(weights / np.maximum(totals, EPSILON))
    return responsibilities


def gaussian_spatial_weights(positions: np.ndarray, sample: AdaptiveSample) -> np.ndarray:
    delta = positions - sample.center
    inverse = np.linalg.pinv(sample.covariance)
    distance = np.einsum("ij,jk,ik->i", delta, inverse, delta)
    return np.exp(-0.5 * np.clip(distance, 0.0, 60.0))


def gaussian_color_weights(colors: np.ndarray, sample: AdaptiveSample) -> np.ndarray:
    delta = colors - sample.color
    distance = np.sum(delta * delta, axis=1) / max(2.0 * sample.color_variance, EPSILON)
    return np.exp(-np.clip(distance, 0.0, 60.0))


def maximization_step(
    values: np.ndarray,
    pixel_positions: np.ndarray,
    samples: list[AdaptiveSample],
    regions: list[np.ndarray],
    responsibilities: list[np.ndarray],
) -> None:
    flattened = values.reshape(-1, values.shape[2])
    for sample, region, gamma in zip(samples, regions, responsibilities, strict=True):
        weight_sum = gamma.sum()
        if weight_sum <= EPSILON:
            continue
        positions = pixel_positions[region]
        colors = flattened[region].astype(np.float64, copy=False)
        sample.center = np.sum(gamma[:, None] * positions, axis=0) / weight_sum
        sample.color = np.sum(gamma[:, None] * colors, axis=0) / weight_sum
        centered = positions - sample.center
        sample.covariance = (centered * gamma[:, None]).T @ centered / weight_sum
        color_delta = colors - sample.color
        sample.color_variance = max(
            float(np.sum(gamma * np.sum(color_delta * color_delta, axis=1)) / weight_sum),
            MIN_COLOR_VARIANCE,
        )


def correction_step(
    samples: list[AdaptiveSample],
    nominal_centers: np.ndarray,
    output_height: int,
    output_width: int,
    ratio_x: float,
    ratio_y: float,
) -> None:
    smooth_centers = [sample.center.copy() for sample in samples]
    for index, _sample in enumerate(samples):
        neighbors = four_neighbors(index, output_height, output_width)
        if neighbors:
            smooth_centers[index] = np.mean(
                [samples[neighbor].center for neighbor in neighbors], axis=0
            )

    for index, sample in enumerate(samples):
        nominal = nominal_centers[index]
        blended = 0.5 * sample.center + 0.5 * smooth_centers[index]
        sample.center = np.array(
            [
                np.clip(blended[0], nominal[0] - ratio_x / 4.0, nominal[0] + ratio_x / 4.0),
                np.clip(blended[1], nominal[1] - ratio_y / 4.0, nominal[1] + ratio_y / 4.0),
            ],
            dtype=np.float64,
        )
        sample.covariance = clamp_covariance(sample.covariance, ratio_x, ratio_y)

    for index, sample in enumerate(samples):
        for neighbor_index in eight_neighbors(index, output_height, output_width):
            neighbor = samples[neighbor_index]
            color_distance = float(np.linalg.norm(sample.color - neighbor.color))
            center_distance = float(np.linalg.norm(sample.center - neighbor.center))
            if color_distance < 0.08 and center_distance > max(ratio_x, ratio_y) * 1.25:
                sample.color_variance *= COLOR_VARIANCE_GROWTH
                neighbor.color_variance *= COLOR_VARIANCE_GROWTH


def clamp_covariance(covariance: np.ndarray, ratio_x: float, ratio_y: float) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    try:
        vectors, singular_values, _ = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return np.array(
            [[max((ratio_x / 3.0) ** 2, 0.05), 0.0], [0.0, max((ratio_y / 3.0) ** 2, 0.05)]]
        )
    lower = max(0.05, min(ratio_x, ratio_y) ** 2 * 0.05)
    upper = max(0.1, max(ratio_x, ratio_y) ** 2 * 0.5)
    clamped = np.clip(singular_values, lower, upper)
    return vectors @ np.diag(clamped) @ vectors.T


def four_neighbors(index: int, height: int, width: int) -> list[int]:
    row, col = divmod(index, width)
    neighbors: list[int] = []
    if row > 0:
        neighbors.append(index - width)
    if row + 1 < height:
        neighbors.append(index + width)
    if col > 0:
        neighbors.append(index - 1)
    if col + 1 < width:
        neighbors.append(index + 1)
    return neighbors


def eight_neighbors(index: int, height: int, width: int) -> list[int]:
    row, col = divmod(index, width)
    neighbors: list[int] = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < height and 0 <= n_col < width:
                neighbors.append(n_row * width + n_col)
    return neighbors


def sample_state(samples: list[AdaptiveSample]) -> np.ndarray:
    return np.concatenate(
        [
            np.concatenate(
                [
                    sample.center,
                    sample.covariance.ravel(),
                    sample.color,
                    np.array([sample.color_variance], dtype=np.float64),
                ]
            )
            for sample in samples
        ]
    )


def has_converged(previous: np.ndarray, current: np.ndarray) -> bool:
    return bool(np.allclose(previous, current, rtol=1.0e-4, atol=1.0e-4))
