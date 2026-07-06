import numpy as np

from scalerack.algorithms.depixelize.boundaries import BoundaryGraph, make_edge_key
from scalerack.algorithms.depixelize.cells import QUARTERS_PER_PIXEL, CellGrid
from scalerack.algorithms.depixelize.similarity_graph import SimilarityGraph
from scalerack.algorithms.depixelize.splines import SplineCurve, quadratic_basis

# Rendering follows the paper's own renderer (section 3.5): truncated Gaussian influence
# from cell centers, blocked across contour lines; regions are rasterized supersampled.
SUPERSAMPLE = 2
GAUSSIAN_SIGMA = 1.0
INFLUENCE_RADIUS = 2.0
INFLUENCE_REACH = int(INFLUENCE_RADIUS)
SECTION_SAMPLES = 8


def render_image(
    rgba: np.ndarray,
    yuva: np.ndarray,
    graph: SimilarityGraph,
    cell_grid: CellGrid,
    boundary_graph: BoundaryGraph,
    curves: list[SplineCurve],
    output_width: int,
    output_height: int,
    supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """Rasterize the smoothed cell geometry and diffuse colors inside diffusion regions."""
    height, width = rgba.shape[:2]
    labels = compute_diffusion_labels(yuva, graph)

    scale_x = output_width / width * supersample
    scale_y = output_height / height * supersample
    grid_height = output_height * supersample
    grid_width = output_width * supersample

    label_map, nearest_y, nearest_x = build_base_maps(
        labels, grid_height, grid_width, scale_x, scale_y
    )
    fill_boundary_cells(label_map, labels, cell_grid, boundary_graph, curves, scale_x, scale_y)
    colors = diffuse_colors(rgba, labels, label_map, nearest_y, nearest_x, scale_x, scale_y)
    supersampled = colors.reshape(output_height, supersample, output_width, supersample, 4)
    return supersampled.mean(axis=(1, 3)).astype(np.float32)


def compute_diffusion_labels(yuva: np.ndarray, graph: SimilarityGraph) -> np.ndarray:
    """Label the connected components of the resolved similarity graph.

    Color influence never crosses visible edges; smooth gradients arise from diffusion
    between connected similar cells inside one region.
    """
    height, width = yuva.shape[:2]
    parent = np.arange(height * width, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for y, x in np.argwhere(graph.east):
        union(y * width + x, y * width + x + 1)
    for y, x in np.argwhere(graph.south):
        union(y * width + x, (y + 1) * width + x)
    for y, x in np.argwhere(graph.se):
        union(y * width + x, (y + 1) * width + x + 1)
    for y, x in np.argwhere(graph.ne):
        union((y + 1) * width + x, y * width + x + 1)

    labels = np.array([find(i) for i in range(height * width)], dtype=np.int64)
    return labels.reshape(height, width)


def build_base_maps(
    labels: np.ndarray,
    grid_height: int,
    grid_width: int,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-pixel region map guaranteeing full coverage before polygon fills refine it."""
    height, width = labels.shape
    ys = np.minimum(((np.arange(grid_height) + 0.5) / scale_y).astype(np.int64), height - 1)
    xs = np.minimum(((np.arange(grid_width) + 0.5) / scale_x).astype(np.int64), width - 1)
    nearest_y = np.broadcast_to(ys[:, None], (grid_height, grid_width))
    nearest_x = np.broadcast_to(xs[None, :], (grid_height, grid_width))
    return labels[nearest_y, nearest_x].copy(), nearest_y, nearest_x


def fill_boundary_cells(
    label_map: np.ndarray,
    labels: np.ndarray,
    cell_grid: CellGrid,
    boundary_graph: BoundaryGraph,
    curves: list[SplineCurve],
    scale_x: float,
    scale_y: float,
) -> None:
    """Overwrite the base map with exact smoothed-polygon coverage near boundaries."""
    sections = build_edge_sections(boundary_graph, curves)
    moved = collect_moved_vertices(boundary_graph, curves)
    width = cell_grid.width

    for cell_index, polygon in enumerate(cell_grid.polygons):
        outline = assemble_cell_outline(polygon, boundary_graph, sections, moved)
        if outline is None:
            continue
        y, x = divmod(cell_index, width)
        region = labels[y, x]
        scaled = outline * (scale_x, scale_y)
        rasterize_polygon(label_map, scaled, region)


def collect_moved_vertices(
    boundary_graph: BoundaryGraph, curves: list[SplineCurve]
) -> dict[tuple[int, int], np.ndarray]:
    """Final position of every boundary-path control point, keyed by its lattice vertex."""
    moved: dict[tuple[int, int], np.ndarray] = {}
    for path, curve in zip(boundary_graph.paths, curves, strict=True):
        for node, position in zip(path.nodes, curve.control_points, strict=True):
            moved[node] = position
    return moved


def build_edge_sections(
    boundary_graph: BoundaryGraph, curves: list[SplineCurve]
) -> dict[tuple[int, int], np.ndarray]:
    """Sampled smooth curve piece for every path segment, ordered along the path."""
    first_half = quadratic_basis(np.linspace(0.0, 0.5, SECTION_SAMPLES))
    second_half = quadratic_basis(np.linspace(0.5, 1.0, SECTION_SAMPLES))
    full_span = quadratic_basis(np.linspace(0.0, 1.0, SECTION_SAMPLES * 2))

    sections: dict[tuple[int, int], np.ndarray] = {}
    for path_index, curve in enumerate(curves):
        points = curve.control_points
        count = len(points)
        segment_count = count if curve.is_closed else count - 1
        for k in range(segment_count):
            open_start = not curve.is_closed and k == 0
            open_end = not curve.is_closed and k == segment_count - 1
            head_basis = full_span if open_start else second_half
            tail_basis = full_span if open_end else first_half
            head = head_basis @ take_span_controls(points, count, curve.is_closed, k)
            tail = tail_basis @ take_span_controls(points, count, curve.is_closed, k + 1)
            sections[(path_index, k)] = np.concatenate((head, tail[1:]))
    return sections


def take_span_controls(points: np.ndarray, count: int, is_closed: bool, span: int) -> np.ndarray:
    """The three control points of one quadratic span, wrapped or end-clamped."""
    indices = (span - 1, span, span + 1)
    if is_closed:
        return points[[index % count for index in indices]]
    return points[[min(max(index, 0), count - 1) for index in indices]]


def assemble_cell_outline(
    polygon: list[tuple[int, int]],
    boundary_graph: BoundaryGraph,
    sections: dict[tuple[int, int], np.ndarray],
    moved: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    """Replace path edges by their smooth sections; None when the plain square suffices."""
    count = len(polygon)
    needs_fill = count != 4 or any(
        qx % QUARTERS_PER_PIXEL or qy % QUARTERS_PER_PIXEL for qx, qy in polygon
    )
    edge_infos = []
    for i in range(count):
        a, b = polygon[i], polygon[(i + 1) % count]
        segment = boundary_graph.edge_segments.get(make_edge_key(a, b))
        edge_infos.append((a, segment))
        if segment is not None or a in moved or b in moved:
            needs_fill = True
    if not needs_fill:
        return None

    pieces: list[np.ndarray] = []
    previous_was_section = False
    for a, segment in edge_infos:
        if segment is None:
            pieces.append(resolve_vertex_position(a, moved)[None, :])
            previous_was_section = False
            continue
        section = sections[segment]
        path = boundary_graph.paths[segment[0]]
        forward = path.nodes[segment[1]] == a
        oriented = section if forward else section[::-1]
        if not previous_was_section:
            pieces.append(resolve_vertex_position(a, moved)[None, :])
        pieces.append(oriented[:-1])
        previous_was_section = True
    return np.concatenate(pieces)


def resolve_vertex_position(
    vertex: tuple[int, int], moved: dict[tuple[int, int], np.ndarray]
) -> np.ndarray:
    """Final position of a lattice vertex: optimized if on a path, original otherwise."""
    position = moved.get(vertex)
    if position is None:
        return np.array(vertex, dtype=np.float64) / QUARTERS_PER_PIXEL
    return position


def rasterize_polygon(label_map: np.ndarray, polygon: np.ndarray, region: int) -> None:
    """Even-odd scanline fill of one polygon onto the supersampled label map."""
    grid_height, grid_width = label_map.shape
    ys = polygon[:, 1]
    row_start = max(0, int(np.ceil(ys.min() - 0.5)))
    row_end = min(grid_height - 1, int(np.floor(ys.max() - 0.5)))
    if row_end < row_start:
        return

    start = polygon
    end = np.roll(polygon, -1, axis=0)
    for row in range(row_start, row_end + 1):
        center = row + 0.5
        crosses = (start[:, 1] <= center) != (end[:, 1] <= center)
        if not np.any(crosses):
            continue
        a = start[crosses]
        b = end[crosses]
        t = (center - a[:, 1]) / (b[:, 1] - a[:, 1])
        xs = np.sort(a[:, 0] + t * (b[:, 0] - a[:, 0]))
        for left, right in zip(xs[0::2], xs[1::2], strict=False):
            column_start = max(0, int(np.ceil(left - 0.5)))
            column_end = min(grid_width - 1, int(np.ceil(right - 0.5)) - 1)
            if column_end >= column_start:
                label_map[row, column_start : column_end + 1] = region


def diffuse_colors(
    rgba: np.ndarray,
    labels: np.ndarray,
    label_map: np.ndarray,
    nearest_y: np.ndarray,
    nearest_x: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Blend cell colors with truncated Gaussian influence confined to each region."""
    height, width = labels.shape
    grid_height, grid_width = label_map.shape
    qx = (np.arange(grid_width) + 0.5) / scale_x
    qy = (np.arange(grid_height) + 0.5) / scale_y
    base_x = np.floor(qx).astype(np.int64)
    base_y = np.floor(qy).astype(np.int64)

    numerator = np.zeros((grid_height, grid_width, 4), dtype=np.float64)
    denominator = np.zeros((grid_height, grid_width), dtype=np.float64)
    inv_two_sigma_squared = 1.0 / (2.0 * GAUSSIAN_SIGMA**2)

    for dy in range(-INFLUENCE_REACH, INFLUENCE_REACH + 1):
        cell_y = base_y + dy
        valid_y = (cell_y >= 0) & (cell_y < height)
        clipped_y = np.clip(cell_y, 0, height - 1)
        distance_y = qy - (cell_y + 0.5)
        for dx in range(-INFLUENCE_REACH, INFLUENCE_REACH + 1):
            cell_x = base_x + dx
            valid_x = (cell_x >= 0) & (cell_x < width)
            clipped_x = np.clip(cell_x, 0, width - 1)
            distance_x = qx - (cell_x + 0.5)

            squared = distance_y[:, None] ** 2 + distance_x[None, :] ** 2
            weight = np.exp(-squared * inv_two_sigma_squared)
            weight *= squared <= INFLUENCE_RADIUS**2
            weight *= valid_y[:, None] & valid_x[None, :]
            weight *= labels[clipped_y[:, None], clipped_x[None, :]] == label_map

            numerator += weight[:, :, None] * rgba[clipped_y[:, None], clipped_x[None, :]]
            denominator += weight

    fallback = rgba[nearest_y, nearest_x]
    safe = denominator > 0.0
    blended = np.where(
        safe[:, :, None], numerator / np.maximum(denominator, 1e-12)[:, :, None], fallback
    )
    return blended
