"""Explicit mixed-cumulant expansion of an exponentially tilted mean."""

from __future__ import annotations

import numpy as np


def prepare_observables(source_energy, target_energy, size, beta):
    """Form dimensionless V from total energies and per-particle target G."""
    source = np.asarray(source_energy, dtype=float)
    target = np.asarray(target_energy, dtype=float)
    if source.shape != target.shape or source.ndim != 1 or source.size == 0:
        raise ValueError("Source and target energies must be nonempty matching vectors")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Energies must be finite")
    if not np.isfinite(size) or size <= 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError("N and beta must be finite positive values")
    delta_U = target - source
    V = -beta * delta_U  # Total energy difference; never delta_U / N.
    G = target / size
    return V, G


def mixed_cumulants(V, G):
    """Return the first five derivatives of the tilted mean at lambda=0."""
    V = np.asarray(V, dtype=float)
    G = np.asarray(G, dtype=float)
    if V.shape != G.shape or V.ndim != 1 or V.size == 0:
        raise ValueError("V and G must be nonempty matching vectors")
    if not np.isfinite(V).all() or not np.isfinite(G).all():
        raise ValueError("V and G must be finite")
    dV = V - V.mean()
    dG = G - G.mean()

    # Centering V and G makes the disconnected terms explicit and reduces
    # cancellation when the mean of V is large.
    v2 = np.mean(dV ** 2)
    v3 = np.mean(dV ** 3)
    vg = np.mean(dV * dG)
    v2g = np.mean(dV ** 2 * dG)
    v3g = np.mean(dV ** 3 * dG) - 3 * v2 * vg
    v4g = (np.mean(dV ** 4 * dG) - 6 * v2 * v2g - 4 * v3 * vg)
    return {
        "kappa_1": float(G.mean()),
        "kappa_VG": float(vg),
        "kappa_VVG": float(v2g),
        "kappa_VVVG": float(v3g),
        "kappa_VVVVG": float(v4g),
    }


def contributions(V, G):
    """Return cumulants, Taylor contributions, and sums through order four."""
    result = mixed_cumulants(V, G)
    terms = (
        result["kappa_1"],
        result["kappa_VG"],
        result["kappa_VVG"] / 2.0,
        result["kappa_VVVG"] / 6.0,
        result["kappa_VVVVG"] / 24.0,
    )
    running = 0.0
    for order, term in enumerate(terms):
        result[f"C{order}"] = term
        running += term
        result[f"G_order{order}"] = running
        if order:
            result[f"abs_C{order}"] = abs(term)
    return result


def tilted_mean(V, G, lambda_=1.0):
    """Compute sum(G exp(lambda V))/sum(exp(lambda V)) with shifted weights."""
    V = np.asarray(V, dtype=float)
    G = np.asarray(G, dtype=float)
    if V.shape != G.shape or V.ndim != 1 or V.size == 0:
        raise ValueError("V and G must be nonempty matching vectors")
    if not np.isfinite(V).all() or not np.isfinite(G).all():
        raise ValueError("V and G must be finite")
    logw = lambda_ * V
    if not np.isfinite(logw).all():
        raise ValueError("lambda V must be finite")
    weights = np.exp(logw - np.max(logw))
    return float(np.dot(weights, G) / weights.sum())


def taylor_coefficients(V, G, max_order):
    """Exact Taylor coefficients C_0..C_max_order of the sample tilted mean.

    F(lambda) = <G exp(lambda V)> / <exp(lambda V)> over the given frames, so
    C_k = kappa(V,...,V,G)/k! with k copies of V, for any order. The series
    of A(lambda) = <dG exp(lambda dV)> is divided by that of
    Z(lambda) = <exp(lambda dV)>. Centering V cancels in F, and centering G
    only moves C_0, so both are centered to limit roundoff in high moments.
    """
    V = np.asarray(V, dtype=float)
    G = np.asarray(G, dtype=float)
    if V.shape != G.shape or V.ndim != 1 or V.size == 0:
        raise ValueError("V and G must be nonempty matching vectors")
    if not np.isfinite(V).all() or not np.isfinite(G).all():
        raise ValueError("V and G must be finite")
    if int(max_order) != max_order or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    dV = V - V.mean()
    dG = G - G.mean()
    a = np.empty(max_order + 1)
    z = np.empty(max_order + 1)
    power = np.ones_like(dV)
    factorial = 1.0
    for k in range(max_order + 1):
        if k:
            power = power * dV
            factorial *= k
        a[k] = np.mean(dG * power) / factorial
        z[k] = np.mean(power) / factorial
    # Power-series division f = a/z, using z[0] = 1.
    f = np.empty(max_order + 1)
    for k in range(max_order + 1):
        f[k] = a[k] - np.dot(z[1:k + 1], f[k - 1::-1][:k]) if k else a[0]
    f[0] += G.mean()
    return f
