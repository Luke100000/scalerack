import numpy as np


def gather_footprint(
    values: np.ndarray,
    row_indices: np.ndarray,
    row_coverage: np.ndarray,
    column_indices: np.ndarray,
    column_coverage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather samples and positive weights from one rectangular footprint."""
    samples = values[row_indices[:, None], column_indices[None, :]].reshape(-1, values.shape[2])
    coverage = (row_coverage[:, None] * column_coverage[None, :]).ravel()
    included = coverage > 0
    return samples[included], coverage[included]
