"""Checks for first-shell RDF feature extraction and volume integration."""

import unittest

import numpy as np

from src.rdf_features import coordination_to_cutoff, first_shell_features


class TestFirstShellFeatures(unittest.TestCase):
    def test_common_cutoff_uses_same_radius_for_distinct_curves(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        density = 0.1
        cutoff = 1.5
        first = coordination_to_cutoff([1.0, 1.0, 1.0], edges, density, cutoff)
        second = coordination_to_cutoff([2.0, 2.0, 2.0], edges, density, cutoff)
        expected = density * 4 * np.pi / 3 * cutoff ** 3
        self.assertAlmostEqual(first, expected)
        self.assertAlmostEqual(second, 2 * expected)

    def test_coordination_integrates_to_minimum_bin_center(self):
        edges = np.arange(0.0, 8.02, 0.02)
        r = (edges[:-1] + edges[1:]) / 2
        # A smooth first peak and trough make the feature locations unambiguous.
        g = 1 + np.exp(-((r - 2.8) / 0.2) ** 2) - 0.25 * np.exp(-((r - 3.5) / 0.2) ** 2)
        density = 0.03
        result = first_shell_features(r, g, edges, density)
        # Independent fine-grid quadrature of the same piecewise-constant RDF.
        fine_r = np.linspace(0, result["first_min_r_A"], 500_001)
        fine_bins = np.searchsorted(edges, fine_r, side="right") - 1
        integrand = 4 * np.pi * density * fine_r ** 2 * g[fine_bins]
        expected = np.trapz(integrand, fine_r)
        self.assertAlmostEqual(result["coordination_number"], expected, delta=1e-4)
        self.assertAlmostEqual(result["first_peak_r_A"], 2.79)
        self.assertAlmostEqual(result["first_min_r_A"], 3.51)


if __name__ == "__main__":
    unittest.main()
