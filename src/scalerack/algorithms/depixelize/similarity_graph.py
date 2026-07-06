from dataclasses import dataclass

import numpy as np

# Dissimilarity thresholds from Kopf & Lischinski 2011, section 3.2 (the hqx criteria).
Y_THRESHOLD = 48.0 / 255.0
U_THRESHOLD = 7.0 / 255.0
V_THRESHOLD = 6.0 / 255.0
# Alpha has no threshold in the paper (opaque art only); treat it as conservatively as luma.
ALPHA_THRESHOLD = Y_THRESHOLD

ISLAND_WEIGHT = 5
SPARSE_WINDOW_RADIUS = 4

RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.5],
        [0.5, -0.419, -0.081],
    ],
    dtype=np.float32,
)


@dataclass
class SimilarityGraph:
    """Pixel connectivity, one boolean array per undirected edge family.

    ``east[y, x]`` connects (y, x)-(y, x+1); ``south[y, x]`` connects (y, x)-(y+1, x);
    ``se[y, x]`` connects (y, x)-(y+1, x+1); ``ne[y, x]`` connects (y+1, x)-(y, x+1).
    """

    east: np.ndarray
    south: np.ndarray
    se: np.ndarray
    ne: np.ndarray


def convert_rgba_to_yuva(rgba: np.ndarray) -> np.ndarray:
    """Map normalized RGBA to YUV plus passthrough alpha."""
    yuv = rgba[:, :, :3] @ RGB_TO_YUV.T
    return np.dstack((yuv, rgba[:, :, 3]))


def pixels_similar(yuva_a: np.ndarray, yuva_b: np.ndarray) -> np.ndarray:
    """Elementwise similarity of two pixel arrays under the per-channel thresholds."""
    difference = np.abs(yuva_a - yuva_b)
    return (
        (difference[..., 0] <= Y_THRESHOLD)
        & (difference[..., 1] <= U_THRESHOLD)
        & (difference[..., 2] <= V_THRESHOLD)
        & (difference[..., 3] <= ALPHA_THRESHOLD)
    )


def build_similarity_graph(yuva: np.ndarray) -> SimilarityGraph:
    """Connect every pixel to its similar 8-neighbors (crossings not yet resolved)."""
    return SimilarityGraph(
        east=pixels_similar(yuva[:, :-1], yuva[:, 1:]),
        south=pixels_similar(yuva[:-1, :], yuva[1:, :]),
        se=pixels_similar(yuva[:-1, :-1], yuva[1:, 1:]),
        ne=pixels_similar(yuva[1:, :-1], yuva[:-1, 1:]),
    )


def compute_valence(graph: SimilarityGraph) -> np.ndarray:
    """Count the connections incident to each pixel."""
    height, width = graph.east.shape[0], graph.south.shape[1]
    valence = np.zeros((height, width), dtype=np.int32)
    valence[:, :-1] += graph.east
    valence[:, 1:] += graph.east
    valence[:-1, :] += graph.south
    valence[1:, :] += graph.south
    valence[:-1, :-1] += graph.se
    valence[1:, 1:] += graph.se
    valence[1:, :-1] += graph.ne
    valence[:-1, 1:] += graph.ne
    return valence


def list_neighbors(graph: SimilarityGraph, y: int, x: int) -> list[tuple[int, int]]:
    """Return the connected neighbors of one pixel."""
    height, width = graph.east.shape[0], graph.south.shape[1]
    neighbors = []
    if x + 1 < width and graph.east[y, x]:
        neighbors.append((y, x + 1))
    if x > 0 and graph.east[y, x - 1]:
        neighbors.append((y, x - 1))
    if y + 1 < height and graph.south[y, x]:
        neighbors.append((y + 1, x))
    if y > 0 and graph.south[y - 1, x]:
        neighbors.append((y - 1, x))
    if y + 1 < height and x + 1 < width and graph.se[y, x]:
        neighbors.append((y + 1, x + 1))
    if y > 0 and x > 0 and graph.se[y - 1, x - 1]:
        neighbors.append((y - 1, x - 1))
    if y > 0 and x + 1 < width and graph.ne[y - 1, x]:
        neighbors.append((y - 1, x + 1))
    if y + 1 < height and x > 0 and graph.ne[y, x - 1]:
        neighbors.append((y + 1, x - 1))
    return neighbors


def resolve_crossings(graph: SimilarityGraph) -> SimilarityGraph:
    """Eliminate crossing diagonals so the graph becomes planar.

    Fully connected 2x2 blocks lose both diagonals outright; contested blocks are decided
    by the paper's curves, sparse-pixels, and islands heuristics, removing both on a tie.
    All votes are measured against the same snapshot so the outcome is order-independent.
    """
    crossing = graph.se & graph.ne
    fully_connected = (
        crossing & graph.east[:-1, :] & graph.east[1:, :] & graph.south[:, :-1] & graph.south[:, 1:]
    )
    graph.se[fully_connected] = False
    graph.ne[fully_connected] = False

    contested = np.argwhere(crossing & ~fully_connected)
    if len(contested) == 0:
        return graph

    snapshot = SimilarityGraph(
        east=graph.east.copy(),
        south=graph.south.copy(),
        se=graph.se.copy(),
        ne=graph.ne.copy(),
    )
    valence = compute_valence(snapshot)

    keep_se = []
    keep_ne = []
    for y, x in contested:
        se_edge = ((y, x), (y + 1, x + 1))
        ne_edge = ((y + 1, x), (y, x + 1))
        se_weight = ISLAND_WEIGHT * edge_has_island(valence, se_edge)
        ne_weight = ISLAND_WEIGHT * edge_has_island(valence, ne_edge)
        se_weight += measure_curve_length(snapshot, valence, se_edge)
        ne_weight += measure_curve_length(snapshot, valence, ne_edge)
        ne_size, se_size = measure_component_sizes(snapshot, y, x)
        # Sparser (smaller) component wins the sparse-pixels vote.
        se_weight += max(0, ne_size - se_size)
        ne_weight += max(0, se_size - ne_size)
        if se_weight > ne_weight:
            keep_se.append((y, x))
        elif ne_weight > se_weight:
            keep_ne.append((y, x))

    graph.se[tuple(np.array(contested).T)] = False
    graph.ne[tuple(np.array(contested).T)] = False
    for y, x in keep_se:
        graph.se[y, x] = True
    for y, x in keep_ne:
        graph.ne[y, x] = True
    return graph


def edge_has_island(valence: np.ndarray, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """True when cutting the edge would strand a single pixel."""
    (ay, ax), (by, bx) = edge
    return valence[ay, ax] == 1 or valence[by, bx] == 1


def measure_curve_length(
    graph: SimilarityGraph,
    valence: np.ndarray,
    edge: tuple[tuple[int, int], tuple[int, int]],
) -> int:
    """Length of the valence-2 curve the edge belongs to (minimum 1)."""
    start_a, start_b = edge
    length = 1
    for anchor, first in ((start_b, start_a), (start_a, start_b)):
        previous, current = anchor, first
        while valence[current] == 2:
            following = [n for n in list_neighbors(graph, *current) if n != previous]
            if len(following) != 1:
                break
            previous, current = current, following[0]
            length += 1
            if (previous, current) in (edge, (start_b, start_a)):
                # The curve closed into a cycle containing the edge itself.
                return length - 1
    return length


def measure_component_sizes(graph: SimilarityGraph, y: int, x: int) -> tuple[int, int]:
    """Sizes of the two diagonals' components inside the 8x8 window around block (y, x)."""
    height, width = graph.east.shape[0], graph.south.shape[1]
    y_min = max(0, y - SPARSE_WINDOW_RADIUS + 1)
    y_max = min(height - 1, y + SPARSE_WINDOW_RADIUS)
    x_min = max(0, x - SPARSE_WINDOW_RADIUS + 1)
    x_max = min(width - 1, x + SPARSE_WINDOW_RADIUS)

    def flood_size(seeds: list[tuple[int, int]]) -> int:
        seen = set(seeds)
        queue = list(seeds)
        while queue:
            node = queue.pop()
            for neighbor in list_neighbors(graph, *node):
                ny, nx = neighbor
                if y_min <= ny <= y_max and x_min <= nx <= x_max and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return len(seen)

    ne_size = flood_size([(y + 1, x), (y, x + 1)])
    se_size = flood_size([(y, x), (y + 1, x + 1)])
    return ne_size, se_size
