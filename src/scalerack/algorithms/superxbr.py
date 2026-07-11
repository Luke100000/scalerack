"""Super xBR, ported from Hyllian's reference implementation.

Port of the single-file C++ reference "Super XBR Scaler",
Copyright (c) 2016 Hyllian - sergiogdb@gmail.com, MIT license
(https://gitlab.com/strandgames/brahman/-/raw/8166fa471a337d5bee8dab80d0d19a779c134036/tools/libxbrtest/superxbr.cpp).
The three passes reproduce the reference exactly, including its sequential
in-place third pass (vectorized here as an order-preserving wavefront).
"""

import numpy as np

from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput, as_image_input

WEIGHT_ONE = 0.129633
WEIGHT_TWO = 0.175068

# diagonal-edge detector weights: full set for passes 1 and 3, only the
# nearest-diagonal terms for pass 2
EDGE_WEIGHTS = (2.0, 1.0, -1.0, 4.0, -1.0, 1.0)
CROSS_EDGE_WEIGHTS = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0)

LUMA_RED = 0.2126
LUMA_GREEN = 0.7152
LUMA_BLUE = 0.0722

# window cells whose min/max bound the result (anti-ringing clamp)
CENTRAL_CELLS = ((1, 1), (2, 1), (1, 2), (2, 2))

Window = dict[tuple[int, int], np.ndarray]


@register(factor=2)
def superxbr(image: ImageInput) -> ImageInput:
    """Enlarge pixel art exactly 2x with Hyllian's Super xBR hybrid interpolation."""
    image_input = as_image_input(image)
    return image_input.from_numpy(expand_superxbr(image_input.rgba()))


def expand_superxbr(values: np.ndarray) -> np.ndarray:
    quantized = np.rint(values.astype(np.float64) * 255.0)
    height, width = quantized.shape[:2]
    result = np.empty((height * 2, width * 2, 4), dtype=np.float64)

    run_diagonal_pass(quantized, result)
    run_cross_pass(result)
    run_refinement_pass(result)
    return (result / 255.0).astype(np.float32)


def run_diagonal_pass(source: np.ndarray, result: np.ndarray) -> None:
    """Pass 1: copy each source pixel into three block cells and fill the
    fourth (the diagonal position) from the 4x4 source window."""
    padded = np.pad(source, ((1, 2), (1, 2), (0, 0)), mode="edge")
    height, width = source.shape[:2]
    # window[(i, j)] follows the reference layout: i = column offset + 1,
    # j = row offset + 1 around the source pixel
    window = {(i, j): padded[j : j + height, i : i + width] for i in range(4) for j in range(4)}
    result[0::2, 0::2] = source
    result[0::2, 1::2] = source
    result[1::2, 0::2] = source
    result[1::2, 1::2] = filter_window(window, EDGE_WEIGHTS, -WEIGHT_ONE, WEIGHT_ONE + 0.5)


def run_cross_pass(result: np.ndarray) -> None:
    """Pass 2: fill the two remaining block cells from diagonally rotated
    windows over the pass-1 grid."""
    output_height, output_width = result.shape[:2]
    snapshot = result.copy()  # interior windows only ever read pass-1 cells

    interior_rows = np.arange(4, output_height - 5, 2)
    interior_columns = np.arange(4, output_width - 5, 2)
    if interior_rows.size and interior_columns.size:
        rows = interior_rows[:, None]
        columns = interior_columns[None, :]
        first = gather_window(snapshot, rows, columns, cross_offsets(0, 0))
        result[rows, columns + 1] = filter_window(
            first, CROSS_EDGE_WEIGHTS, -WEIGHT_TWO, WEIGHT_TWO + 0.5
        )
        second = gather_window(snapshot, rows, columns, cross_offsets(1, -1))
        result[rows + 1, columns] = filter_window(
            second, CROSS_EDGE_WEIGHTS, -WEIGHT_TWO, WEIGHT_TWO + 0.5, clamp_window=first
        )

    # border anchors clamp their reads, which can touch cells this pass already
    # wrote; replay them sequentially in the reference's raster order
    interior_row_set = set(interior_rows.tolist())
    interior_column_set = set(interior_columns.tolist())
    for row in range(0, output_height, 2):
        for column in range(0, output_width, 2):
            if row in interior_row_set and column in interior_column_set:
                continue
            anchor_row = np.array([[row]])
            anchor_column = np.array([[column]])
            first = gather_window(result, anchor_row, anchor_column, cross_offsets(0, 0))
            result[row, column + 1] = filter_window(
                first, CROSS_EDGE_WEIGHTS, -WEIGHT_TWO, WEIGHT_TWO + 0.5
            )[0, 0]
            second = gather_window(result, anchor_row, anchor_column, cross_offsets(1, -1))
            result[row + 1, column] = filter_window(
                second, CROSS_EDGE_WEIGHTS, -WEIGHT_TWO, WEIGHT_TWO + 0.5, clamp_window=first
            )[0, 0]


def cross_offsets(row_shift: int, column_shift: int) -> dict[tuple[int, int], tuple[int, int]]:
    """The diagonally rotated pass-2 window: cell (i, j) reads the pixel at
    (sx - sy + row_shift, sx + sy + column_shift) with sx = i - 1, sy = j - 1."""
    return {
        (i, j): ((i - 1) - (j - 1) + row_shift, (i - 1) + (j - 1) + column_shift)
        for i in range(4)
        for j in range(4)
    }


def run_refinement_pass(result: np.ndarray) -> None:
    """Pass 3: re-filter every cell in the reference's reverse raster order.

    The scan updates in place and each cell reads neighbors at row/column
    offsets -2..+1, so cells sharing the same value of 3*row + column are
    mutually independent and every dependency lands in an earlier wavefront —
    processing wavefronts in descending order reproduces the scan exactly.
    """
    output_height, output_width = result.shape[:2]
    offsets = {(i, j): (j - 2, i - 2) for i in range(4) for j in range(4)}
    all_rows = np.arange(output_height)
    for wavefront in range(3 * (output_height - 1) + output_width - 1, -1, -1):
        columns = wavefront - 3 * all_rows
        valid = (columns >= 0) & (columns < output_width)
        if not valid.any():
            continue
        rows = all_rows[valid]
        columns = columns[valid]
        window = gather_window(result, rows, columns, offsets)
        result[rows, columns] = filter_window(window, EDGE_WEIGHTS, -WEIGHT_ONE, WEIGHT_ONE + 0.5)


def gather_window(
    grid: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    offsets: dict[tuple[int, int], tuple[int, int]],
) -> Window:
    """Sample the 4x4 window per cell with the reference's clamped indexing."""
    height, width = grid.shape[:2]
    return {
        cell: grid[
            np.clip(rows + row_offset, 0, height - 1),
            np.clip(columns + column_offset, 0, width - 1),
        ]
        for cell, (row_offset, column_offset) in offsets.items()
    }


def filter_window(
    window: Window,
    edge_weights: tuple[float, ...],
    outer_weight: float,
    inner_weight: float,
    clamp_window: Window | None = None,
) -> np.ndarray:
    """Pick the interpolant along the weaker diagonal, clamp against the
    central pixels (anti-ringing), and quantize like the reference."""
    edge_strength = diagonal_edge(window, edge_weights)
    falling = outer_weight * (window[0, 3] + window[3, 0]) + inner_weight * (
        window[1, 2] + window[2, 1]
    )
    rising = outer_weight * (window[0, 0] + window[3, 3]) + inner_weight * (
        window[1, 1] + window[2, 2]
    )
    value = np.where((edge_strength <= 0.0)[..., None], falling, rising)

    bounds = clamp_window if clamp_window is not None else window
    central = np.stack([bounds[cell] for cell in CENTRAL_CELLS])
    value = np.clip(value, central.min(axis=0), central.max(axis=0))
    return np.clip(np.ceil(value), 0.0, 255.0)


def diagonal_edge(window: Window, weights: tuple[float, ...]) -> np.ndarray:
    """The reference's diagonal edge detector on the window's luma."""
    luma = {
        # summed strictly left to right so ties resolve exactly like the reference
        cell: LUMA_RED * pixels[..., 0] + LUMA_GREEN * pixels[..., 1] + LUMA_BLUE * pixels[..., 2]
        for cell, pixels in window.items()
    }

    def difference(first: tuple[int, int], second: tuple[int, int]) -> np.ndarray:
        return np.abs(luma[first] - luma[second])

    falling_weight = (
        weights[0]
        * (
            difference((0, 2), (1, 1))
            + difference((1, 1), (2, 0))
            + difference((1, 3), (2, 2))
            + difference((2, 2), (3, 1))
        )
        + weights[1] * (difference((0, 3), (1, 2)) + difference((2, 1), (3, 0)))
        + weights[2] * (difference((0, 3), (2, 1)) + difference((1, 2), (3, 0)))
        + weights[3] * difference((1, 2), (2, 1))
        + weights[4] * (difference((0, 2), (2, 0)) + difference((1, 3), (3, 1)))
        + weights[5] * (difference((0, 1), (1, 0)) + difference((2, 3), (3, 2)))
    )
    rising_weight = (
        weights[0]
        * (
            difference((0, 1), (1, 2))
            + difference((1, 2), (2, 3))
            + difference((1, 0), (2, 1))
            + difference((2, 1), (3, 2))
        )
        + weights[1] * (difference((0, 0), (1, 1)) + difference((2, 2), (3, 3)))
        + weights[2] * (difference((0, 0), (2, 2)) + difference((1, 1), (3, 3)))
        + weights[3] * difference((1, 1), (2, 2))
        + weights[4] * (difference((1, 0), (3, 2)) + difference((0, 1), (2, 3)))
        + weights[5] * (difference((0, 2), (1, 3)) + difference((2, 0), (3, 1)))
    )
    return falling_weight - rising_weight
