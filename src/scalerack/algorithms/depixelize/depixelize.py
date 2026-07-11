from scalerack.algorithms.depixelize.boundaries import extract_boundaries
from scalerack.algorithms.depixelize.cells import build_cell_grid
from scalerack.algorithms.depixelize.optimize import ITERATIONS, optimize_splines
from scalerack.algorithms.depixelize.render import SUPERSAMPLE, render_image
from scalerack.algorithms.depixelize.similarity_graph import (
    build_similarity_graph,
    convert_rgba_to_yuva,
    resolve_crossings,
)
from scalerack.algorithms.depixelize.splines import fit_splines
from scalerack.algorithms.registry import register
from scalerack.image_io import ImageInput, as_image_input


@register()
def depixelize(
    image: ImageInput,
    factor: float | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    iterations: int = ITERATIONS,
    supersample: int = SUPERSAMPLE,
) -> ImageInput:
    """Vectorize pixel art into smooth outlines (Kopf-Lischinski 2011) at any target size.

    Intended for hard-edged, limited-palette pixel art at enlarging factors; factors at or
    below 1 are accepted but produce odd-looking results. The paper's optional harmonic-map
    cell relaxation is omitted.

    Args:
        iterations: Curve-relaxation sweeps; higher values smooth staircases more
            aggressively at the cost of drifting from the source shapes, 0 disables.
        supersample: Antialiasing oversampling factor per output axis; higher values give
            softer edges and cost proportionally more time and memory.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations!r}")
    if supersample < 1:
        raise ValueError(f"supersample must be at least 1, got {supersample!r}")

    image_input = as_image_input(image)
    output_width, output_height = image_input.get_target_dimensions(width, height, factor)

    rgba = image_input.rgba()
    input_height, input_width = rgba.shape[:2]
    yuva = convert_rgba_to_yuva(rgba)

    graph = resolve_crossings(build_similarity_graph(yuva))
    cell_grid = build_cell_grid(graph, input_height, input_width)
    boundary_graph = extract_boundaries(cell_grid, yuva)
    curves = fit_splines(boundary_graph)
    optimize_splines(curves, iterations)
    result = render_image(
        rgba,
        yuva,
        graph,
        cell_grid,
        boundary_graph,
        curves,
        output_width,
        output_height,
        supersample,
    )
    return image_input.from_numpy(result)
