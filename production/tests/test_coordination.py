"""Checks for per-frame coordination and the block bootstrap."""

import unittest

import numpy as np

from src.coordination import (block_labels, bootstrap_differences,
                              coordination_differences, frame_coordination)


class TestCoordination(unittest.TestCase):
    def test_simple_cubic_lattice_has_six_neighbours(self):
        # A periodic 4x4x4 simple cubic lattice: six neighbours at spacing a.
        a = 3.0
        grid = np.arange(4) * a
        positions = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
        lengths = np.full(3, 4 * a)
        self.assertEqual(frame_coordination(positions, np.zeros(3), lengths, 1.1 * a), 6.0)
        # Second shell at sqrt(2) a adds twelve more.
        self.assertEqual(frame_coordination(positions, np.zeros(3), lengths, 1.5 * a), 18.0)

    def test_blocks_do_not_straddle_runs(self):
        runs = np.repeat([1, 2, 3], [10, 2, 12])
        labels = block_labels(runs, 3)
        # 10 // 3 = 3 blocks, a short run is one block, 12 // 3 = 4 blocks.
        self.assertEqual(len(np.unique(labels)), 8)
        for label in np.unique(labels):
            self.assertEqual(len(np.unique(runs[labels == label])), 1)

    def test_identical_ensembles_give_zero_differences(self):
        rng = np.random.default_rng(0)
        cn = rng.normal(5, 0.1, 200)
        sample = {"cn": cn, "weight": np.full(200, 1 / 200), "run": np.repeat([1, 2], 100)}
        estimate = coordination_differences(sample, sample)
        for value in estimate.values():
            self.assertAlmostEqual(value, 0.0)
        replicates = bootstrap_differences(sample, sample, 10, 50, rng)
        self.assertEqual(len(replicates["delta_direct"]), 50)


if __name__ == "__main__":
    unittest.main()
