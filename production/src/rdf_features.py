"""First-shell RDF features and shell-volume coordination integrals."""

import numpy as np
from scipy.signal import find_peaks


# One first-shell cutoff for every system size and every curve. The direct mW
# first minimum is shallow, and its detected bin wanders between 3.49 and
# 3.53 A with noise; one 0.01 A shift changes coordination by ~0.04. The value
# is the mean of the six per-size direct mW minima (3.503 A) on the 0.02 A
# bin grid, so coordination differences across N reflect only g(r) and density.
COMMON_CUTOFF_A = 3.50


def coordination_to_cutoff(g, edges, density, cutoff):
    """Integrate a bin-averaged RDF to a common physical radius."""
    g = np.asarray(g, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if g.ndim != 1 or edges.shape != (len(g) + 1,):
        raise ValueError("RDF and edge arrays have inconsistent shapes")
    if not np.isfinite(g).all() or not np.isfinite(edges).all():
        raise ValueError("RDF and radial edges must be finite")
    if (np.diff(edges) <= 0).any() or not np.isclose(edges[0], 0):
        raise ValueError("Radial edges must increase from zero")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Number density must be positive and finite")
    if not np.isfinite(cutoff) or not edges[0] < cutoff <= edges[-1]:
        raise ValueError("Cutoff must lie within the RDF radial range")

    # Include complete shells below the cutoff and the physical fraction of
    # its containing shell. This also handles a cutoff exactly on a bin edge.
    index = min(int(np.searchsorted(edges, cutoff, side="right") - 1), len(g) - 1)
    full_volumes = 4 * np.pi / 3 * (edges[1:index + 1] ** 3
                                    - edges[:index] ** 3)
    partial_volume = 4 * np.pi / 3 * (cutoff ** 3 - edges[index] ** 3)
    return float(density * (np.dot(g[:index], full_volumes)
                            + g[index] * partial_volume))


def first_shell_features(r, g, edges, density, peak_prominence=0.05,
                         minimum_prominence=0.03):
    """Return the first prominent peak, following minimum, and coordination.

    The RDF is treated as constant within each original radial shell. The
    final shell is integrated only from its inner edge to the minimum's
    reported position (the bin center).
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if r.ndim != 1 or g.shape != r.shape or edges.shape != (len(r) + 1,):
        raise ValueError("r, g, and radial edges have inconsistent shapes")
    if not np.isfinite(r).all() or not np.isfinite(g).all():
        raise ValueError("RDF values and radii must be finite")
    if not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        raise ValueError("Radial edges must be finite and increasing")
    if not np.allclose(r, (edges[:-1] + edges[1:]) / 2, atol=1e-9, rtol=0):
        raise ValueError("RDF radii must match the radial-bin centers")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Number density must be positive and finite")

    peaks, _ = find_peaks(g, prominence=peak_prominence, distance=8)
    if not len(peaks):
        raise ValueError("No first RDF peak found")
    peak_index = int(peaks[0])
    minima, _ = find_peaks(-g, prominence=minimum_prominence, distance=8)
    following_minima = minima[minima > peak_index]
    if not len(following_minima):
        raise ValueError("No minimum found after the first RDF peak")
    minimum_index = int(following_minima[0])

    # Integrating g(r) r^2 over each radial shell is equivalent to multiplying
    # the bin-averaged RDF by its shell volume. Stop at the minimum bin center.
    coordination = coordination_to_cutoff(g, edges, density, r[minimum_index])
    return {
        "first_peak_r_A": r[peak_index],
        "first_peak_g": g[peak_index],
        "first_min_r_A": r[minimum_index],
        "coordination_number": coordination,
    }
