"""Verify paired bootstrap calculations and signed/absolute percentile handling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cumulant_truncation import bootstrap_residuals, summarize_residuals


class FixedDraws:
    def __init__(self, indices):
        self.indices = iter(indices)

    def integers(self, low, high, size):
        return np.array(next(self.indices))


class TruncationBootstrapTests(unittest.TestCase):
    def test_each_draw_uses_its_own_full_mean_and_paired_frames(self):
        V = np.array([-0.3, 0.1, 0.5])
        A = np.array([-0.4, 0.2, 0.9])
        indices = [[0, 0, 2], [1, 2, 2]]
        _, draws = bootstrap_residuals(V, A, 2, FixedDraws(indices), max_order=12)
        original_full = np.sum(A * np.exp(V)) / np.exp(V).sum()
        for b, index in enumerate(indices):
            v, a = V[index], A[index]
            # Independently checked C0, C1 and C2 and unshifted exponential ratio.
            full = np.sum(a * np.exp(v)) / np.exp(v).sum()
            dv, da = v - v.mean(), a - a.mean()
            partial = np.cumsum([a.mean(), np.mean(dv * da), np.mean(dv**2 * da) / 2])
            np.testing.assert_allclose(draws[b, :3], partial - full, atol=2e-16)
            self.assertGreater(abs(full - original_full), 0.01)

    def test_constant_frame_draw_has_no_truncation_error_at_any_order(self):
        _, draws = bootstrap_residuals([0, 1, 2], [1, 4, 9], 2,
                                      FixedDraws([[0, 0, 0], [2, 2, 2]]))
        np.testing.assert_array_equal(draws, np.zeros((2, 13)))

    def test_absolute_percentiles_are_not_absolute_signed_percentiles(self):
        draws = np.array([[-4.0], [-1.0], [2.0], [3.0]])
        summary = summarize_residuals(np.array([-2.0]), draws)
        self.assertEqual(summary["original_epsilon"][0], 2.0)
        self.assertEqual(summary["bootstrap_median_signed_difference"][0], 0.5)
        self.assertEqual(summary["bootstrap_median_epsilon"][0], 2.5)
        self.assertAlmostEqual(summary["bootstrap_epsilon_p16"][0], 1.48)
        self.assertAlmostEqual(summary["bootstrap_std_signed_difference"][0],
                               np.std(draws[:, 0], ddof=1))


if __name__ == "__main__":
    unittest.main()
