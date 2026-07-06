import itertools
from dataclasses import dataclass

import numpy as np

from scalerack.algorithms.depixelize.boundaries import (
    BoundaryGraph,
    BoundaryPath,
    EdgeKey,
    Vertex,
    make_edge_key,
)
from scalerack.algorithms.depixelize.cells import QUARTERS_PER_PIXEL

# Figure 7 corner patterns as consecutive control-point offsets on the quarter-pixel lattice
# (see specs/002-depixelize-pixel-art/geometry-notes.md). Matched in every rotation,
# reflection, and traversal direction; matched segments are excluded from smoothing.
CORNER_PATTERNS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, -3), (3, -3), (4, 0)),
    ((-1, -1), (1, -3), (3, -3), (4, 0)),
    ((-1, -1), (1, -3), (3, -3), (5, -1)),
    ((0, 0), (0, -4), (4, -4), (4, 0)),
    ((1, -1), (-1, -3), (-3, -3), (-3, -1), (-1, 1)),
)


@dataclass
class SplineCurve:
    """A quadratic B-spline over one boundary path.

    ``smooth_segments[k]`` is False where the curve between control points k and k+1 is
    excluded from the smoothness energy (corner patterns, image-border runs);
    ``free_nodes`` marks control points the optimizer may move.
    """

    control_points: np.ndarray
    initial_points: np.ndarray
    is_closed: bool
    smooth_segments: np.ndarray
    free_nodes: np.ndarray


def quadratic_basis(ts: np.ndarray) -> np.ndarray:
    """Quadratic B-spline span basis weights for parameters ``ts``, shape (T, 3)."""
    return np.stack(
        (0.5 * (1.0 - ts) ** 2, -(ts**2) + ts + 0.5, 0.5 * ts**2),
        axis=-1,
    )


def build_pattern_variants() -> frozenset[tuple[tuple[int, int], ...]]:
    """Delta sequences of every corner pattern under rotation, mirroring, and reversal."""
    variants: set[tuple[tuple[int, int], ...]] = set()
    for pattern in CORNER_PATTERNS:
        deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(pattern)]
        for mirrored in (deltas, [(-dx, dy) for dx, dy in deltas]):
            rotated = mirrored
            for _ in range(4):
                variants.add(tuple(rotated))
                variants.add(tuple((-dx, -dy) for dx, dy in reversed(rotated)))
                rotated = [(-dy, dx) for dx, dy in rotated]
    return frozenset(variants)


PATTERN_VARIANTS = build_pattern_variants()
PATTERN_LENGTHS = frozenset(len(variant) for variant in PATTERN_VARIANTS)


def fit_splines(boundary_graph: BoundaryGraph) -> list[SplineCurve]:
    """Turn boundary paths into spline curves with corner exclusions and fixed endpoints."""
    curves = []
    for path in boundary_graph.paths:
        points = np.array(path.nodes, dtype=np.float64) / QUARTERS_PER_PIXEL
        segment_count = len(path.nodes) if path.is_closed else len(path.nodes) - 1
        smooth = np.ones(segment_count, dtype=bool)
        mark_corner_segments(path, smooth)

        free = np.ones(len(path.nodes), dtype=bool)
        mask_border_segments(path, boundary_graph.border_edges, smooth, free)
        if not path.is_closed:
            free[0] = False
            free[-1] = False
            adjust_junction_endpoint(path, points, 0, boundary_graph.junction_continuations)
            adjust_junction_endpoint(path, points, -1, boundary_graph.junction_continuations)

        curves.append(
            SplineCurve(
                control_points=points,
                initial_points=points.copy(),
                is_closed=path.is_closed,
                smooth_segments=smooth,
                free_nodes=free,
            )
        )
    return curves


def mask_border_segments(
    path: BoundaryPath,
    border_edges: set[EdgeKey],
    smooth: np.ndarray,
    free: np.ndarray,
) -> None:
    """Pin border geometry: the image edge is a hard clip, not a curve to optimize."""
    nodes = path.nodes
    count = len(nodes)
    for k in range(len(smooth)):
        a, b = nodes[k], nodes[(k + 1) % count]
        if make_edge_key(a, b) in border_edges:
            smooth[k] = False
            free[k] = False
            free[(k + 1) % count] = False


def mark_corner_segments(path: BoundaryPath, smooth: np.ndarray) -> None:
    """Exclude segments matching any corner pattern from the smoothness energy."""
    nodes = path.nodes
    count = len(nodes)
    segment_count = len(smooth)
    for length in PATTERN_LENGTHS:
        if segment_count < length:
            continue
        window_starts = range(segment_count if path.is_closed else segment_count - length + 1)
        for start in window_starts:
            deltas = []
            for offset in range(length):
                a = nodes[(start + offset) % count]
                b = nodes[(start + offset + 1) % count]
                deltas.append((b[0] - a[0], b[1] - a[1]))
            if tuple(deltas) in PATTERN_VARIANTS:
                for offset in range(length):
                    smooth[(start + offset) % segment_count] = False


def adjust_junction_endpoint(
    path: BoundaryPath,
    points: np.ndarray,
    end: int,
    continuations: dict[Vertex, tuple[Vertex, Vertex]],
) -> None:
    """Move a T-junction endpoint onto the curve continuing through the junction.

    A quadratic B-spline with controls (A, J, B) passes nearest J at (A + 6J + B) / 8.
    """
    junction = path.nodes[end]
    continuation = continuations.get(junction)
    if continuation is None:
        return
    a, b = continuation
    adjusted = (
        np.array(a, dtype=np.float64)
        + 6.0 * np.array(junction, dtype=np.float64)
        + np.array(b, dtype=np.float64)
    ) / 8.0
    points[end] = adjusted / QUARTERS_PER_PIXEL
