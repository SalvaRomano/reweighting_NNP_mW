"""Per-frame coordination numbers and block-bootstrap errors of their differences."""

import numpy as np
from scipy.spatial import cKDTree


def frame_coordination(positions, lower_bounds, lengths, cutoff):
    """Mean number of minimum-image neighbours within cutoff for one frame."""
    positions = np.asarray(positions, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must contain at least two 3D particles")
    if not 0 < cutoff <= 0.5 * np.min(lengths):
        raise ValueError("cutoff must be positive and at most half the box")
    wrapped = (positions - lower_bounds) % lengths
    tree = cKDTree(wrapped, boxsize=lengths)
    # Each unordered pair within the cutoff is one neighbour for both atoms.
    return 2.0 * len(tree.query_pairs(cutoff)) / len(positions)


def weighted_mean(values, weights):
    """Mean of values with non-negative weights normalized to one."""
    weights = np.asarray(weights, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


def block_labels(runs, block_length):
    """Split every run into contiguous blocks of about block_length frames.

    Frames must be ordered in time within each run. The returned label is
    unique per (run, block), so blocks never straddle two trajectories. Runs
    differ strongly in length, so the block length (not the block count) is
    fixed; a run shorter than block_length forms a single block.
    """
    runs = np.asarray(runs)
    labels = np.empty(len(runs), dtype=int)
    next_label = 0
    for run in np.unique(runs):
        index = np.flatnonzero(runs == run)
        n_blocks = max(1, len(index) // block_length)
        for chunk in np.array_split(index, n_blocks):
            labels[chunk] = next_label
            next_label += 1
    return labels


def resample_blocks(labels, rng):
    """Frame indices of one block-bootstrap replicate.

    All blocks of one ensemble (pooled over its runs) are drawn with
    replacement. Contiguous blocks preserve the time correlation of
    consecutive frames inside each block.
    """
    blocks = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in chosen])


def coordination_differences(nnp, mw):
    """Signed coordination differences for direct and reweighted estimates.

    Each argument is a dict with per-frame arrays "cn" and "weight", where
    "weight" reweights that ensemble to the other potential.
    """
    cn_nnp = float(np.mean(nnp["cn"]))
    cn_mw = float(np.mean(mw["cn"]))
    cn_nnp_to_mw = weighted_mean(nnp["cn"], nnp["weight"])
    cn_mw_to_nnp = weighted_mean(mw["cn"], mw["weight"])
    return {
        "delta_direct": cn_nnp - cn_mw,
        "delta_NNP_to_mW": cn_nnp_to_mw - cn_mw,
        "delta_mW_to_NNP": cn_mw_to_nnp - cn_nnp,
    }


def bootstrap_differences(nnp, mw, block_length, n_boot, rng):
    """Block-bootstrap replicates of coordination_differences.

    NNP and mW are independent simulations, so they are resampled
    independently. Weights are renormalized inside each replicate, which
    propagates the weight degeneracy (small N_eff) into the error bars.
    """
    labels = {"NNP": block_labels(nnp["run"], block_length),
              "mW": block_labels(mw["run"], block_length)}
    replicates = []
    for _ in range(n_boot):
        i = resample_blocks(labels["NNP"], rng)
        j = resample_blocks(labels["mW"], rng)
        replicates.append(coordination_differences(
            {key: nnp[key][i] for key in ("cn", "weight")},
            {key: mw[key][j] for key in ("cn", "weight")}))
    return {key: np.array([rep[key] for rep in replicates])
            for key in replicates[0]}
