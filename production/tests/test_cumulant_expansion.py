"""Check explicit mixed cumulants against numerical tilted-mean derivatives."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cumulant_expansion import (contributions, prepare_observables,
                                    taylor_coefficients, tilted_mean)


def numerical_derivatives(V, G, step):
    """Use central finite differences of an independently implemented F(lambda)."""
    def F(lambda_):
        logw = lambda_ * V
        weights = np.exp(logw - logw.max())
        return np.dot(weights, G) / weights.sum()

    fm2, fm1, f0, fp1, fp2 = (F(k * step) for k in (-2, -1, 0, 1, 2))
    return np.array([
        f0,
        (fm2 - 8 * fm1 + 8 * fp1 - fp2) / (12 * step),
        (-fm2 + 16 * fm1 - 30 * f0 + 16 * fp1 - fp2) / (12 * step ** 2),
        (-fm2 + 2 * fm1 - 2 * fp1 + fp2) / (2 * step ** 3),
        (fm2 - 4 * fm1 + 6 * f0 - 4 * fp1 + fp2) / step ** 4,
    ])


class CumulantExpansionTests(unittest.TestCase):
    def test_total_energy_difference_and_per_particle_observable(self):
        source = np.array([-20.0, -21.0, -22.0])
        target = np.array([-19.8, -20.7, -21.9])
        V, G = prepare_observables(source, target, size=10, beta=2.0)
        np.testing.assert_allclose(V, -2.0 * (target - source))
        np.testing.assert_allclose(G, target / 10.0)

    def test_cumulants_match_numerical_derivatives_through_fourth_order(self):
        # Asymmetric V and nonlinear G make all mixed-cumulant corrections active.
        V = np.array([-1.3, -0.6, -0.2, 0.1, 0.4, 1.2, 2.1])
        G = np.array([-0.2, 0.3, -0.1, 0.8, 0.0, 1.4, -0.7])
        result = contributions(V, G)
        h = 0.12
        coarse = numerical_derivatives(V, G, h)
        fine = numerical_derivatives(V, G, h / 2)
        # Third and fourth central differences have O(h^2) truncation error.
        extrapolated = (4 * fine - coarse) / 3
        expected = np.array([
            result["kappa_1"], result["kappa_VG"], result["kappa_VVG"],
            result["kappa_VVVG"], result["kappa_VVVVG"],
        ])
        np.testing.assert_allclose(extrapolated, expected, rtol=2e-4, atol=2e-5)
        for order, factorial in enumerate((1, 1, 2, 6, 24)):
            self.assertAlmostEqual(result[f"C{order}"], expected[order] / factorial)
            self.assertAlmostEqual(
                result[f"G_order{order}"],
                sum(result[f"C{k}"] for k in range(order + 1)),
            )

    def test_large_constant_shift_in_V_does_not_change_results(self):
        V = np.array([-0.8, -0.3, 0.1, 0.7, 1.4])
        G = np.array([0.2, -0.4, 0.6, 0.1, 1.2])
        base = contributions(V, G)
        shifted = contributions(V + 1000.0, G)
        for order in range(5):
            self.assertAlmostEqual(base[f"C{order}"], shifted[f"C{order}"], places=10)
        self.assertAlmostEqual(tilted_mean(V, G), tilted_mean(V + 1000.0, G))

    def test_taylor_coefficients_match_explicit_cumulants(self):
        V = np.array([-1.3, -0.6, -0.2, 0.1, 0.4, 1.2, 2.1])
        G = np.array([-0.2, 0.3, -0.1, 0.8, 0.0, 1.4, -0.7])
        coefficients = taylor_coefficients(V, G, 12)
        result = contributions(V, G)
        np.testing.assert_allclose(
            coefficients[:5], [result[f"C{k}"] for k in range(5)], rtol=1e-12, atol=1e-15)

    def test_taylor_series_sums_to_tilted_mean_when_convergent(self):
        # Small spread of V keeps lambda=1 well inside the convergence radius.
        rng = np.random.default_rng(3)
        V = 0.5 * rng.normal(size=400)
        G = 0.4 * V + rng.normal(size=400)
        coefficients = taylor_coefficients(V, G, 30)
        self.assertAlmostEqual(coefficients.sum(), tilted_mean(V, G), places=12)
        shifted = taylor_coefficients(V + 1000.0, G, 30)
        np.testing.assert_allclose(shifted, coefficients, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
