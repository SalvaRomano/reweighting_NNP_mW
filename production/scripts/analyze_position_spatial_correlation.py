#!/usr/bin/env python3
"""Calculate the positional radial distribution function from the N=64 mW run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rdf import frame_rdf, radial_edges  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ROOT / "results/energy_autocorrelation_N64/water.lammpstrj",
    )
    parser.add_argument("--dr", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/energy_autocorrelation_N64/position_rdf.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures/Structural/position_spatial_correlation_N64.png",
    )
    return parser.parse_args()


def read_dump(path):
    """Yield step, box, and ID-sorted coordinates from an orthorhombic dump."""
    with path.open(encoding="utf-8") as handle:
        while marker := handle.readline():
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Unexpected trajectory marker: {marker.strip()}")
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Missing atom-count header at step {step}")
            count = int(handle.readline())
            if handle.readline().split()[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError(f"Missing box header at step {step}")
            bounds = np.array([[float(value) for value in handle.readline().split()]
                               for _ in range(3)])
            header = handle.readline().split()[2:]
            required = {"id", "x", "y", "z"}
            if not required.issubset(header):
                raise ValueError(f"Missing coordinate columns at step {step}")
            id_col = header.index("id")
            xyz_cols = [header.index(axis) for axis in "xyz"]
            coordinates = np.empty((count, 3), dtype=float)
            for _ in range(count):
                fields = handle.readline().split()
                atom_id = int(fields[id_col])
                coordinates[atom_id - 1] = [float(fields[col]) for col in xyz_cols]
            lower = bounds[:, 0]
            lengths = bounds[:, 1] - lower
            yield step, lower, lengths, coordinates


def main():
    args = parse_args()
    total = None
    total_sq = None
    edges = None
    steps = []
    reference_lengths = None

    for step, lower, lengths, coordinates in read_dump(args.trajectory):
        if edges is None:
            reference_lengths = lengths
            edges = radial_edges(args.dr, 0.5 * np.min(lengths))
            total = np.zeros(len(edges) - 1)
            total_sq = np.zeros(len(edges) - 1)
        elif not np.allclose(lengths, reference_lengths, rtol=0.0, atol=1e-8):
            raise ValueError("The trajectory box changes between frames")
        values = frame_rdf(coordinates, lower, lengths, edges)
        total += values
        total_sq += values**2
        steps.append(step)

    if not steps:
        raise ValueError(f"No frames found in {args.trajectory}")
    mean = total / len(steps)
    # Frame-to-frame SEM is descriptive; adjacent frames are time-correlated.
    variance = (total_sq - len(steps) * mean**2) / (len(steps) - 1)
    sem = np.sqrt(np.maximum(variance, 0.0) / len(steps))
    radii = 0.5 * (edges[:-1] + edges[1:])

    output = pd.DataFrame({"r_A": radii, "g_r": mean, "frame_sem": sem})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.12g")

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.plot(radii, mean, color="#2166ac", lw=1.6, label=r"Position correlation $g(r)$")
    ax.fill_between(radii, mean - sem, mean + sem, color="#2166ac", alpha=0.2,
                    label="Frame SEM")
    ax.axhline(1.0, color="0.45", lw=0.8, ls="--")
    ax.set(xlim=(0.0, edges[-1]), xlabel=r"Separation $r$ ($\mathrm{\AA}$)",
           ylabel=r"Radial distribution function $g(r)$",
           title="N=64 mW positional spatial correlation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=300, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"frames={len(steps)} dr_A={args.dr} r_max_A={edges[-1]:.12g}")
    print(f"Saved {args.output} and {args.figure.with_suffix('.*')}")


if __name__ == "__main__":
    main()
