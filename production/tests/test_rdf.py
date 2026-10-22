"""Check minimum-image pair counting and per-frame RDF normalization."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rdf import frame_rdf, radial_edges


class RdfTests(unittest.TestCase):
    def test_periodic_pair_is_counted_once_with_frame_density(self):
        positions = np.array([[0.1, 0.2, 0.3], [9.9, 0.2, 0.3]])
        lengths = np.array([10.0, 10.0, 10.0])
        edges = np.array([0.0, 0.5, 1.0])
        g = frame_rdf(positions, np.zeros(3), lengths, edges)
        shell_volume = 4 * np.pi / 3 * 0.5 ** 3
        expected = 1 / (0.5 * 2 * (2 / 1000) * shell_volume)
        np.testing.assert_allclose(g, [expected, 0.0])

    def test_radial_grid_ends_at_half_box_without_tiny_last_bin(self):
        r_max = 19.7205782451 / 2
        edges = radial_edges(0.02, r_max)
        self.assertAlmostEqual(edges[-1], r_max)
        np.testing.assert_allclose(np.diff(edges[:-1]), 0.02, atol=1e-12)
        self.assertGreater(np.diff(edges)[-1], 0.01)


if __name__ == "__main__":
    unittest.main()
