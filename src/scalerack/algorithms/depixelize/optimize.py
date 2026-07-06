import numpy as np

from scalerack.algorithms.depixelize.splines import SplineCurve, quadratic_basis

# The paper leaves the relaxation parameters unpublished; the offset radius follows
# libdepixelize and the iteration count was calibrated against the published teaser render.
# The paper's random walk is replaced by a fixed compass stencil and red-black sweeps to
# keep the output deterministic.
OFFSET_RADIUS = 0.125
ITERATIONS = 2
SAMPLES_PER_SPAN = 7
MAX_DISPLACEMENT = 0.5
POSITIONAL_POWER = 4
# Balances squared-radian curvature units against the quartic positional term.
CURVATURE_WEIGHT = 40.0

DIAGONAL = float(np.sqrt(0.5))
CANDIDATE_OFFSETS = OFFSET_RADIUS * np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [DIAGONAL, DIAGONAL],
        [0.0, 1.0],
        [-DIAGONAL, DIAGONAL],
        [-1.0, 0.0],
        [-DIAGONAL, -DIAGONAL],
        [0.0, -1.0],
        [DIAGONAL, -DIAGONAL],
    ],
    dtype=np.float64,
)


def optimize_splines(curves: list[SplineCurve], iterations: int = ITERATIONS) -> None:
    """Relax control points against curvature plus positional energy, in place."""
    layout = flatten_curves(curves)
    if layout is None:
        return
    points, original, free, parity, neighbor_indices, span_smooth = layout

    basis = quadratic_basis(np.linspace(0.0, 1.0, SAMPLES_PER_SPAN))
    for _ in range(iterations):
        for color in (0, 1):
            active = free & (parity == color)
            if not np.any(active):
                continue
            relax_nodes(points, original, active, neighbor_indices, span_smooth, basis)

    offset = 0
    for curve in curves:
        count = len(curve.control_points)
        curve.control_points[:] = points[offset : offset + count]
        offset += count


def flatten_curves(
    curves: list[SplineCurve],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Concatenate all curves into flat arrays with per-node neighbor indices."""
    points_parts = []
    free_parts = []
    parity_parts = []
    neighbor_parts = []
    span_smooth_parts = []
    base = 0
    for curve in curves:
        count = len(curve.control_points)
        points_parts.append(curve.control_points)
        free_parts.append(curve.free_nodes)
        parity_parts.append(np.arange(count) % 2)

        local = np.arange(count)
        offsets = np.stack([local - 2, local - 1, local, local + 1, local + 2], axis=1)
        if curve.is_closed:
            offsets %= count
        else:
            offsets = np.clip(offsets, 0, count - 1)
        neighbor_parts.append(offsets + base)

        span_smooth_parts.append(compute_span_smoothness(curve))
        base += count

    if base == 0:
        return None
    points = np.concatenate(points_parts).astype(np.float64, copy=True)
    return (
        points,
        points.copy(),
        np.concatenate(free_parts),
        np.concatenate(parity_parts),
        np.concatenate(neighbor_parts),
        np.concatenate(span_smooth_parts),
    )


def compute_span_smoothness(curve: SplineCurve) -> np.ndarray:
    """Whether the span centered at each node participates in the smoothness integral."""
    count = len(curve.control_points)
    segments = curve.smooth_segments
    smooth = np.ones(count, dtype=bool)
    for node in range(count):
        if curve.is_closed:
            before = segments[(node - 1) % count]
            after = segments[node % count]
        else:
            before = segments[node - 1] if node - 1 >= 0 else True
            after = segments[node] if node < len(segments) else True
        smooth[node] = bool(before) and bool(after)
    return smooth


def relax_nodes(
    points: np.ndarray,
    original: np.ndarray,
    active: np.ndarray,
    neighbor_indices: np.ndarray,
    span_smooth: np.ndarray,
    basis: np.ndarray,
) -> None:
    """One red-black half-sweep: every active node tries the candidate stencil, keeps the best."""
    indices = np.flatnonzero(active)
    window = points[neighbor_indices[indices]]
    node_original = original[indices]
    span_eligibility = span_smooth[neighbor_indices[indices][:, 1:4]]

    best_energy = None
    best_positions = None
    for offset in CANDIDATE_OFFSETS:
        candidate = points[indices] + offset
        smoothness = measure_curvature(window, candidate, basis, span_eligibility)
        positional_distance = np.linalg.norm(candidate - node_original, axis=1)
        energy = smoothness + positional_distance**POSITIONAL_POWER
        if best_energy is None:
            best_energy = energy
            best_positions = candidate
        else:
            better = energy < best_energy
            best_energy = np.where(better, energy, best_energy)
            best_positions = np.where(better[:, None], candidate, best_positions)

    displacement = best_positions - node_original
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)
    scale = np.minimum(1.0, MAX_DISPLACEMENT / np.maximum(distance, 1e-12))
    points[indices] = node_original + displacement * scale


def measure_curvature(
    window: np.ndarray,
    candidate: np.ndarray,
    basis: np.ndarray,
    span_eligibility: np.ndarray,
) -> np.ndarray:
    """Squared sampled turning of the three spans influenced by each candidate position.

    The paper integrates |curvature|, but that measure cannot tell a smooth arc from the
    same turn concentrated into a kink; squaring the sampled angles prefers even arcs.
    """
    a2, a1, b1, b2 = window[:, 0], window[:, 1], window[:, 3], window[:, 4]
    spans = np.stack(
        (
            np.stack((a2, a1, candidate), axis=1),
            np.stack((a1, candidate, b1), axis=1),
            np.stack((candidate, b1, b2), axis=1),
        ),
        axis=1,
    )
    samples = np.einsum("tc,nsck->nstk", basis, spans)
    count = samples.shape[2]
    joined = np.concatenate((samples[:, 0], samples[:, 1, 1:], samples[:, 2, 1:]), axis=1)
    directions = np.diff(joined, axis=1)
    cross = (
        directions[:, :-1, 0] * directions[:, 1:, 1] - directions[:, :-1, 1] * directions[:, 1:, 0]
    )
    dot = (directions[:, :-1] * directions[:, 1:]).sum(axis=2)
    angles = np.abs(np.arctan2(cross, dot))

    directions_per_span = count - 1
    direction_span = np.concatenate([np.full(directions_per_span, s) for s in range(3)])
    angle_mask = np.take(span_eligibility, direction_span[:-1], axis=1) & np.take(
        span_eligibility, direction_span[1:], axis=1
    )
    return CURVATURE_WEIGHT * (angles**2 * angle_mask).sum(axis=1)
