"""Paired configuration bootstrap of empirical Taylor truncation residuals."""

import numpy as np

from .cumulant_expansion import taylor_coefficients, tilted_mean


def signed_truncation_residual(V, A, max_order=12):
    """Recompute partial sums and the full mean from the same configurations."""
    partial = np.cumsum(taylor_coefficients(V, A, max_order))
    return partial - tilted_mean(V, A)


def bootstrap_residuals(V, A, repetitions, rng, max_order=12):
    """Return signed residuals; retain their sign until after each calculation."""
    V, A = np.asarray(V, dtype=float), np.asarray(A, dtype=float)
    original = signed_truncation_residual(V, A, max_order)
    if repetitions < 2:
        raise ValueError("At least two bootstrap realizations are required")
    draws = np.empty((repetitions, max_order + 1))
    for b in range(repetitions):
        indices = rng.integers(0, len(V), len(V))
        # This paired draw preserves correlations among coefficients and full mean.
        draws[b] = signed_truncation_residual(V[indices], A[indices], max_order)
    if not np.isfinite(draws).all():
        raise ValueError("Non-finite bootstrap residuals")
    return original, draws


def summarize_residuals(original, draws):
    """Return per-order summaries; absolute-value quantiles are computed separately."""
    epsilon = np.abs(draws)
    summary = {
        "original_signed_difference": original,
        "original_epsilon": np.abs(original),
        "bootstrap_mean_signed_difference": draws.mean(axis=0),
        "bootstrap_median_signed_difference": np.median(draws, axis=0),
        "bootstrap_std_signed_difference": draws.std(axis=0, ddof=1),
        "bootstrap_median_epsilon": np.median(epsilon, axis=0),
    }
    for label, percentile in (("p2_5", 2.5), ("p16", 16), ("p50", 50),
                              ("p84", 84), ("p97_5", 97.5)):
        summary[f"bootstrap_signed_difference_{label}"] = np.percentile(
            draws, percentile, axis=0)
        if label != "p50":
            summary[f"bootstrap_epsilon_{label}"] = np.percentile(
                epsilon, percentile, axis=0)
    return summary
