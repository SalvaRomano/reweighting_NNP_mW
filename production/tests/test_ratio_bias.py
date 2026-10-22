"""Unit tests for self-normalized ratio bias utilities."""

import unittest

import numpy as np

from src.ratio_bias import (leading_order_terms, population_ratio_moments,
                            weighted_inverse_n_fit)


class RatioBiasTests(unittest.TestCase):
    def test_weighted_inverse_n_fit_recovers_coefficient(self):
        n = np.array([50.0, 100.0, 200.0, 500.0])
        sem = np.array([2e-5, 1e-5, 8e-6, 4e-6])
        coefficient, coefficient_sem = weighted_inverse_n_fit(
            n, 0.003 / n, sem)
        expected_sem = 1 / np.sqrt(np.sum((1 / n) ** 2 / sem ** 2))
        self.assertAlmostEqual(coefficient, 0.003)
        self.assertAlmostEqual(coefficient_sem, expected_sem)

    def test_population_moments_match_explicit_ddof_zero_formulas(self):
        g = np.array([-1.0, 0.5, 2.0, 4.0])
        logw = np.array([-3.0, -1.0, 0.5, 2.0])
        result = population_ratio_moments(g, logw)
        w = np.exp(logw - logw.max())
        x = g * w
        mu_x, mu_y = x.mean(), w.mean()
        var_y = np.mean((w - mu_y) ** 2)
        cov_xy = np.mean((x - mu_x) * (w - mu_y))
        expected_b = mu_x * var_y / mu_y ** 3 - cov_xy / mu_y ** 2
        self.assertAlmostEqual(result["theta_full"], x.sum() / w.sum())
        self.assertAlmostEqual(result["varY"], np.var(w, ddof=0))
        self.assertAlmostEqual(result["covXY"], cov_xy)
        self.assertAlmostEqual(result["B_theory"], expected_b)

    def test_constant_logweight_offset_does_not_change_results(self):
        g = np.array([1.0, 2.0, 5.0])
        logw = np.array([-2.0, 0.0, 1.0])
        first = population_ratio_moments(g, logw)
        second = population_ratio_moments(g, logw + 1234.5)
        self.assertAlmostEqual(first["theta_full"], second["theta_full"])
        self.assertAlmostEqual(first["B_theory"], second["B_theory"])
        np.testing.assert_allclose(first["weights"], second["weights"])

    def test_leading_order_identity_and_noise(self):
        rng = np.random.default_rng(4)
        g = rng.normal(size=200)
        logw = 0.8 * g + rng.normal(size=200)
        terms = leading_order_terms(g, logw)
        # B = -(M/N_eff) (theta_w2 - theta) holds exactly for population moments.
        self.assertAlmostEqual(
            terms["B_theory"], -terms["inefficiency"] * terms["delta_theta_w2"])
        w = np.exp(logw - logw.max())
        theta = np.dot(w, g) / w.sum()
        sigma = np.sqrt(np.mean((w * (g - theta)) ** 2)) / w.mean()
        self.assertAlmostEqual(terms["sigma"], sigma)


if __name__ == "__main__":
    unittest.main()
