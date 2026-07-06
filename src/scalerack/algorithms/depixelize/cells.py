from dataclasses import dataclass

import numpy as np

from scalerack.algorithms.depixelize.similarity_graph import SimilarityGraph

# Corner states: which diagonal of the surrounding 2x2 block crosses this grid corner.
CORNER_PLAIN = 0
CORNER_NW_SE = 1
CORNER_NE_SW = 2

# Roles a pixel plays relative to one of its corners (position within the corner's 2x2 block).
ROLE_NW = 0
ROLE_NE = 1
ROLE_SW = 2
ROLE_SE = 3

QUARTERS_PER_PIXEL = 4

# Vertex emission templates in quarter-pixel offsets from the corner, ordered along the
# clockwise (TL, TR, BR, BL) polygon winding. Cells cut by a diagonal lose the corner to a
# single chamfer vertex; the two diagonally connected cells share the perpendicular edge
# through the corner (see specs/002-depixelize-pixel-art/geometry-notes.md).
CORNER_TEMPLATES: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (CORNER_PLAIN, ROLE_NW): ((0, 0),),
    (CORNER_PLAIN, ROLE_NE): ((0, 0),),
    (CORNER_PLAIN, ROLE_SW): ((0, 0),),
    (CORNER_PLAIN, ROLE_SE): ((0, 0),),
    (CORNER_NW_SE, ROLE_NE): ((1, -1),),
    (CORNER_NW_SE, ROLE_SW): ((-1, 1),),
    (CORNER_NW_SE, ROLE_NW): ((1, -1), (-1, 1)),
    (CORNER_NW_SE, ROLE_SE): ((-1, 1), (1, -1)),
    (CORNER_NE_SW, ROLE_NW): ((-1, -1),),
    (CORNER_NE_SW, ROLE_SE): ((1, 1),),
    (CORNER_NE_SW, ROLE_NE): ((1, 1), (-1, -1)),
    (CORNER_NE_SW, ROLE_SW): ((-1, -1), (1, 1)),
}


@dataclass
class CellGrid:
    """Reshaped pixel cells: one clockwise polygon per pixel on the quarter-pixel lattice."""

    polygons: list[list[tuple[int, int]]]
    height: int
    width: int


def classify_corners(graph: SimilarityGraph, height: int, width: int) -> np.ndarray:
    """State of every grid corner: plain, or crossed by one resolved diagonal."""
    states = np.zeros((height + 1, width + 1), dtype=np.int8)
    interior = states[1:-1, 1:-1]
    interior[graph.se] = CORNER_NW_SE
    interior[graph.ne] = CORNER_NE_SW
    return states


def build_cell_grid(graph: SimilarityGraph, height: int, width: int) -> CellGrid:
    """Assemble every pixel's polygon by walking its corners clockwise."""
    states = classify_corners(graph, height, width)
    polygons: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            corner_specs = (
                (x, y, ROLE_SE),
                (x + 1, y, ROLE_SW),
                (x + 1, y + 1, ROLE_NW),
                (x, y + 1, ROLE_NE),
            )
            polygon: list[tuple[int, int]] = []
            for cx, cy, role in corner_specs:
                state = int(states[cy, cx])
                for dx, dy in CORNER_TEMPLATES[(state, role)]:
                    polygon.append((cx * QUARTERS_PER_PIXEL + dx, cy * QUARTERS_PER_PIXEL + dy))
            polygons.append(polygon)
    return CellGrid(polygons=polygons, height=height, width=width)
