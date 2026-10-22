"""Finite-sample bias utilities for a self-normalized ratio estimator."""

import numpy as np


def population_ratio_moments(g, logw):
    """Compute fixed-shift population moments and the leading bias coefficient."""
    g = np.asarray(g, dtype=float)
    logw = np.asarray(logw, dtype=float)
    if g.ndim != 1 or logw.shape != g.shape or not len(g):
        raise ValueError("g and logw must be nonempty one-dimensional arrays")
    if not np.isfinite(g).all() or not np.isfinite(logw).all():
        raise ValueError("g and logw must be finite")

    shift = float(np.max(logw))
    weights = np.exp(logw - shift)
    x = g * weights
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(weights))
    var_y = float(np.mean((weights - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (weights - mu_y)))
    theta = mu_x / mu_y
    bias_coefficient = mu_x * var_y / mu_y ** 3 - cov_xy / mu_y ** 2
    neff = float(weights.sum() ** 2 / np.dot(weights, weights))
    return {
        "shift": shift,
        "weights": weights,
        "x": x,
        "muX": mu_x,
        "muY": mu_y,
        "varY": var_y,
        "covXY": cov_xy,
        "theta_full": theta,
        "B_theory": float(bias_coefficient),
        "neff_full": neff,
    }


def bootstrap_ratio(x, y, sample_size, repetitions, rng, max_indices=2_000_000):
    """Bootstrap a ratio of sample means in bounded-memory batches."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("x and y must be nonempty one-dimensional arrays")
    if sample_size <= 0 or repetitions <= 1 or max_indices < sample_size:
        raise ValueError("Invalid bootstrap size, repetitions, or batch limit")

    estimates = np.empty(repetitions, dtype=float)
    batch_repetitions = max(1, max_indices // sample_size)
    for start in range(0, repetitions, batch_repetitions):
        stop = min(start + batch_repetitions, repetitions)
        indices = rng.integers(0, len(x), size=(stop - start, sample_size))
        estimates[start:stop] = x[indices].sum(axis=1) / y[indices].sum(axis=1)
    return estimates


def weighted_inverse_n_fit(sample_sizes, biases, standard_errors):
    """Fit bias = B/n by weighted least squares without an intercept."""
    n = np.asarray(sample_sizes, dtype=float)
    bias = np.asarray(biases, dtype=float)
    sem = np.asarray(standard_errors, dtype=float)
    if n.ndim != 1 or bias.shape != n.shape or sem.shape != n.shape or not len(n):
        raise ValueError("Fit inputs must be nonempty one-dimensional arrays")
    if (n <= 0).any() or (sem <= 0).any() or not np.isfinite([n, bias, sem]).all():
        raise ValueError("Fit inputs must be finite with positive n and SEM")
    predictor = 1.0 / n
    inverse_variance = 1.0 / sem ** 2
    information = np.sum(inverse_variance * predictor ** 2)
    coefficient = np.sum(inverse_variance * predictor * bias) / information
    return float(coefficient), float(np.sqrt(1.0 / information))


def leading_order_terms(g, logw):
    """Leading-order (1/n) bias and noise quantities of the ratio estimator.

    Uses the identity B = -(M/N_eff) * (theta_w2 - theta), where theta_w2 is
    the observable averaged with squared weights. M/N_eff measures the weight
    inefficiency, and theta_w2 - theta the correlation of G with the weights.
    The asymptotic standard error at n samples is sigma/sqrt(n) with
    sigma = sqrt(E[Z^2])/mu_Y and Z = w (G - theta).
    """
    moments = population_ratio_moments(g, logw)
    g = np.asarray(g, dtype=float)
    weights = moments["weights"]
    theta = moments["theta_full"]
    squared = weights ** 2
    theta_w2 = float(np.dot(squared, g) / squared.sum())
    z = weights * (g - theta)
    return {
        "theta_full": theta,
        "B_theory": moments["B_theory"],
        "neff_full": moments["neff_full"],
        "inefficiency": len(g) / moments["neff_full"],
        "theta_w2": theta_w2,
        "delta_theta_w2": theta_w2 - theta,
        "sigma": float(np.sqrt(np.mean(z ** 2)) / moments["muY"]),
    }
