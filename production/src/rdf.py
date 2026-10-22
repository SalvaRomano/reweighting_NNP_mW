"""Frame-level radial distribution functions for periodic single-species boxes."""

import numpy as np
from scipy.spatial import cKDTree


def radial_edges(dr, r_max):
    """Make dr-spaced bins ending exactly at the requested radius."""
    if not np.isfinite(dr) or not np.isfinite(r_max) or dr <= 0 or r_max <= dr:
        raise ValueError("dr and r_max must be finite, positive, with r_max > dr")
    edges = np.arange(0.0, r_max, dr)
    # Avoid a nearly empty last shell when half the box length is just above
    # an integer multiple of dr. Only the terminal shell may differ from dr.
    if r_max - edges[-1] < dr / 2:
        edges[-1] = r_max
    else:
        edges = np.append(edges, r_max)
    return edges


def frame_rdf(positions, lower_bounds, lengths, edges):
    """Count unordered minimum-image pairs and normalize one configuration."""
    positions = np.asarray(positions, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must contain at least two 3D particles")
    if lower_bounds.shape != (3,) or lengths.shape != (3,):
        raise ValueError("Box bounds must each have three components")
    if not np.isfinite(positions).all() or not np.isfinite(lengths).all():
        raise ValueError("Positions and box lengths must be finite")
    if (lengths <= 0).any() or not np.isfinite(lower_bounds).all():
        raise ValueError("Box lengths must be positive and bounds finite")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0:
        raise ValueError("Radial edges must start at zero")
    if not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        raise ValueError("Radial edges must be finite and increasing")
    if edges[-1] > 0.5 * np.min(lengths) + 1e-10:
        raise ValueError("r_max exceeds half the shortest box length")

    count = len(positions)
    wrapped = (positions - lower_bounds) % lengths
    tree = cKDTree(wrapped, boxsize=lengths)
    ordered_counts = tree.count_neighbors(tree, edges, cumulative=False)
    # Same-tree counts include N self-pairs at r=0 and count every distinct
    # pair twice. The positive-radius bins therefore divide by two.
    if ordered_counts[0] != count or np.any(ordered_counts[1:] % 2):
        raise ValueError("Unexpected self-pair or odd periodic pair count")
    histogram = ordered_counts[1:] / 2.0
    shell_volumes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = count / np.prod(lengths)
    return histogram / (0.5 * count * density * shell_volumes)
