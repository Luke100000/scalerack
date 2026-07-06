import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from scalerack.algorithms.depixelize.cells import CellGrid
from scalerack.algorithms.depixelize.similarity_graph import (
    ALPHA_THRESHOLD,
    U_THRESHOLD,
    V_THRESHOLD,
    Y_THRESHOLD,
)

# A visible edge is a shading edge when the meeting cells stay within this per-channel
# YUV(+alpha) distance; otherwise it is a contour edge (paper section 3.3).
SHADING_LIMIT = 100.0 / 255.0

Vertex = tuple[int, int]
EdgeKey = tuple[Vertex, Vertex]


@dataclass
class BoundaryPath:
    nodes: list[Vertex]
    is_closed: bool


@dataclass
class BoundaryGraph:
    """Visible color boundaries linked into maximal paths on the quarter-pixel lattice."""

    paths: list[BoundaryPath]
    edge_segments: dict[EdgeKey, tuple[int, int]] = field(default_factory=dict)
    junction_continuations: dict[Vertex, tuple[Vertex, Vertex]] = field(default_factory=dict)
    border_edges: set[EdgeKey] = field(default_factory=set)


def make_edge_key(a: Vertex, b: Vertex) -> EdgeKey:
    return (a, b) if a <= b else (b, a)


def extract_boundaries(cell_grid: CellGrid, yuva: np.ndarray) -> BoundaryGraph:
    """Find visible edges, link them into paths, and resolve T-junctions.

    Image-border edges join the graph as contour edges so region outlines corner-cut at
    border contacts instead of terminating flat against the image edge.
    """
    flat_yuva = yuva.reshape(-1, yuva.shape[2])
    edge_cells = collect_edge_cells(cell_grid)

    adjacency: dict[Vertex, list[Vertex]] = {}
    shading: dict[EdgeKey, bool] = {}
    border_edges: set[EdgeKey] = set()
    for (a, b), cells in edge_cells.items():
        if len(cells) == 1:
            border_edges.add((a, b))
            shading[(a, b)] = False
        else:
            difference = np.abs(flat_yuva[cells[0]] - flat_yuva[cells[1]])
            if not is_visible(difference):
                continue
            shading[(a, b)] = bool(np.all(difference <= SHADING_LIMIT))
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    paths = link_paths(adjacency)
    graph = BoundaryGraph(paths=paths, border_edges=border_edges)
    merge_t_junctions(graph, adjacency, shading, border_edges)

    for path_index, path in enumerate(graph.paths):
        for segment_index, (a, b) in enumerate(iterate_segments(path)):
            graph.edge_segments[make_edge_key(a, b)] = (path_index, segment_index)
    return graph


def is_visible(channel_difference: np.ndarray) -> bool:
    """Cell edges are visible where the similarity thresholds are exceeded."""
    return bool(
        channel_difference[0] > Y_THRESHOLD
        or channel_difference[1] > U_THRESHOLD
        or channel_difference[2] > V_THRESHOLD
        or channel_difference[3] > ALPHA_THRESHOLD
    )


def collect_edge_cells(cell_grid: CellGrid) -> dict[EdgeKey, list[int]]:
    """Map every cell-polygon edge to the cells sharing it."""
    edge_cells: dict[EdgeKey, list[int]] = {}
    for cell_index, polygon in enumerate(cell_grid.polygons):
        count = len(polygon)
        for i in range(count):
            a, b = polygon[i], polygon[(i + 1) % count]
            if a == b:
                continue
            edge_cells.setdefault(make_edge_key(a, b), []).append(cell_index)
    return edge_cells


def link_paths(adjacency: dict[Vertex, list[Vertex]]) -> list[BoundaryPath]:
    """Chain visible edges into maximal paths ending only at junctions or closing into loops."""
    visited: set[EdgeKey] = set()
    paths: list[BoundaryPath] = []

    def walk(start: Vertex, first: Vertex) -> list[Vertex]:
        nodes = [start, first]
        visited.add(make_edge_key(start, first))
        while len(adjacency[nodes[-1]]) == 2:
            previous, current = nodes[-2], nodes[-1]
            following = next(n for n in adjacency[current] if n != previous)
            edge = make_edge_key(current, following)
            if edge in visited:
                break
            visited.add(edge)
            nodes.append(following)
        return nodes

    junctions = sorted(v for v, others in adjacency.items() if len(others) != 2)
    for vertex in junctions:
        for neighbor in sorted(adjacency[vertex]):
            if make_edge_key(vertex, neighbor) not in visited:
                paths.append(BoundaryPath(nodes=walk(vertex, neighbor), is_closed=False))

    for vertex in sorted(adjacency):
        for neighbor in sorted(adjacency[vertex]):
            if make_edge_key(vertex, neighbor) not in visited:
                nodes = walk(vertex, neighbor)
                # Remaining edges only occur in pure valence-2 loops; drop the repeated vertex.
                closed = nodes[0] == nodes[-1]
                paths.append(BoundaryPath(nodes=nodes[:-1] if closed else nodes, is_closed=closed))
    return paths


def iterate_segments(path: BoundaryPath) -> list[tuple[Vertex, Vertex]]:
    pairs = list(itertools.pairwise(path.nodes))
    if path.is_closed and len(path.nodes) > 2:
        pairs.append((path.nodes[-1], path.nodes[0]))
    return pairs


def merge_t_junctions(
    graph: BoundaryGraph,
    adjacency: dict[Vertex, list[Vertex]],
    shading: dict[EdgeKey, bool],
    border_edges: set[EdgeKey],
) -> None:
    """Combine two of the three splines meeting at each T-junction into one curve."""
    for junction in sorted(v for v, others in adjacency.items() if len(others) == 3):
        ends = find_path_ends(graph.paths, junction)
        if len(ends) != 3:
            continue
        merge_pair = choose_merge_pair(graph.paths, junction, ends, shading, border_edges)
        if merge_pair is None:
            continue
        apply_merge(graph, junction, ends[merge_pair[0]], ends[merge_pair[1]])


def find_path_ends(paths: list[BoundaryPath], junction: Vertex) -> list[tuple[int, int]]:
    """Locate open-path ends at a junction as (path_index, end) with end 0 or -1."""
    ends = []
    for index, path in enumerate(paths):
        if path.is_closed:
            continue
        if path.nodes[0] == junction:
            ends.append((index, 0))
        if path.nodes[-1] == junction:
            ends.append((index, -1))
    return ends


def choose_merge_pair(
    paths: list[BoundaryPath],
    junction: Vertex,
    ends: list[tuple[int, int]],
    shading: dict[EdgeKey, bool],
    border_edges: set[EdgeKey],
) -> tuple[int, int] | None:
    """Pick which two branches continue through the junction.

    One shading edge among three means the two contour edges continue; any other mix is
    settled by the pair whose angle is closest to straight. At the image border a contour
    branch joins one of the two border runs (a silhouette corner the spline then cuts);
    letting the border merge with itself would pin the artwork flat against the edge.
    """
    branch_nodes = [first_node_from(paths[p], end) for p, end in ends]
    branch_border = [make_edge_key(junction, node) in border_edges for node in branch_nodes]
    branch_shading = [shading.get(make_edge_key(junction, node), False) for node in branch_nodes]

    if sum(branch_border) == 2:
        interior = branch_border.index(False)
        if branch_shading[interior]:
            border_indices = [i for i, is_border in enumerate(branch_border) if is_border]
            return border_indices[0], border_indices[1]
        candidate_pairs = [(min(interior, i), max(interior, i)) for i in range(3) if i != interior]
        return straightest_pair(junction, branch_nodes, candidate_pairs)

    if sum(branch_shading) == 1:
        contour_indices = [i for i, is_shading in enumerate(branch_shading) if not is_shading]
        return contour_indices[0], contour_indices[1]

    all_pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    return straightest_pair(junction, branch_nodes, all_pairs)


def straightest_pair(
    junction: Vertex,
    branch_nodes: list[Vertex],
    candidate_pairs: list[tuple[int, int]],
) -> tuple[int, int] | None:
    best_pair = None
    best_deviation = None
    for i, j in candidate_pairs:
        deviation = abs(math.pi - angle_between(junction, branch_nodes[i], branch_nodes[j]))
        if best_deviation is None or deviation < best_deviation:
            best_deviation = deviation
            best_pair = (i, j)
    return best_pair


def first_node_from(path: BoundaryPath, end: int) -> Vertex:
    return path.nodes[1] if end == 0 else path.nodes[-2]


def angle_between(junction: Vertex, a: Vertex, b: Vertex) -> float:
    va = (a[0] - junction[0], a[1] - junction[1])
    vb = (b[0] - junction[0], b[1] - junction[1])
    dot = va[0] * vb[0] + va[1] * vb[1]
    cross = va[0] * vb[1] - va[1] * vb[0]
    return abs(math.atan2(cross, dot))


def apply_merge(
    graph: BoundaryGraph,
    junction: Vertex,
    end_a: tuple[int, int],
    end_b: tuple[int, int],
) -> None:
    """Join two path ends into one curve running through the junction."""
    path_a_index, which_a = end_a
    path_b_index, which_b = end_b
    path_a = graph.paths[path_a_index]

    if path_a_index == path_b_index:
        # Both ends of the same path meet here: it closes into a loop through the junction.
        graph.junction_continuations[junction] = (path_a.nodes[-2], path_a.nodes[1])
        path_a.nodes = path_a.nodes[:-1]
        path_a.is_closed = True
        return

    path_b = graph.paths[path_b_index]
    nodes_a = path_a.nodes if which_a == -1 else list(reversed(path_a.nodes))
    nodes_b = path_b.nodes if which_b == 0 else list(reversed(path_b.nodes))
    merged = nodes_a + nodes_b[1:]
    graph.junction_continuations[junction] = (merged[len(nodes_a) - 2], merged[len(nodes_a)])

    graph.paths[path_a_index] = BoundaryPath(nodes=merged, is_closed=False)
    del graph.paths[path_b_index]
