#!/usr/bin/env python3
"""Plot signed N=256 RDF residuals and report first-shell features."""

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
from src.rdf import radial_edges  # noqa: E402
from src.rdf_features import (COMMON_CUTOFF_A, coordination_to_cutoff,  # noqa: E402
                              first_shell_features)


RDF_COLUMNS = ("g_NNP", "g_mW", "g_NNP_to_mW", "g_mW_to_NNP")
N = 256
DR = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "analysis/rdf_N256.csv")
    parser.add_argument("--trajectory", type=Path,
                        default=ROOT / "lammps/mW_NVT/N_256/water.lammpstrj")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/rdf_N256")
    parser.add_argument("--figure-prefix", type=Path,
                        default=ROOT / "figures/Structural/rdf_N256_residuals")
    parser.add_argument("--cutoff", type=float, default=COMMON_CUTOFF_A,
                        help=f"Common coordination cutoff in A (default: {COMMON_CUTOFF_A})")
    return parser.parse_args()


def first_box_lengths(path, expected_particles=N):
    """Read box lengths from the first orthorhombic frame in a LAMMPS dump."""
    with path.open(encoding="utf-8") as handle:
        if handle.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError(f"Invalid LAMMPS dump: {path}")
        handle.readline()  # timestep
        if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"Missing atom-count header in {path}")
        count = int(handle.readline())
        if count != expected_particles:
            raise ValueError(f"Expected {expected_particles} particles, found {count} in {path}")
        if handle.readline().split() != ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"]:
            raise ValueError(f"Expected a periodic orthorhombic box in {path}")
        bounds = [list(map(float, handle.readline().split())) for _ in range(3)]
    if any(len(pair) != 2 for pair in bounds):
        raise ValueError(f"Invalid box bounds in {path}")
    lengths = np.array([upper - lower for lower, upper in bounds])
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError(f"Invalid box lengths in {path}")
    return lengths


def main():
    args = parse_args()
    rdf = pd.read_csv(args.input)
    required = {"r", *RDF_COLUMNS}
    if not required.issubset(rdf.columns):
        raise ValueError(f"Missing RDF columns: {sorted(required - set(rdf.columns))}")
    if rdf[list(required)].isna().any().any():
        raise ValueError("RDF input contains missing values")

    lengths = first_box_lengths(args.trajectory)
    density = N / np.prod(lengths)
    edges = radial_edges(DR, np.min(lengths) / 2)
    r = rdf["r"].to_numpy(dtype=float)
    if len(r) != len(edges) - 1:
        raise ValueError("RDF bins do not match the N=256 trajectory box")

    residuals = pd.DataFrame({
        "r": r,
        "delta_g_direct": rdf["g_NNP"] - rdf["g_mW"],
        "delta_g_NNP_to_mW": rdf["g_NNP_to_mW"] - rdf["g_mW"],
        "delta_g_mW_to_NNP": rdf["g_mW_to_NNP"] - rdf["g_NNP"],
    })
    args.results_dir.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(args.results_dir / "residuals.csv", index=False)

    # Peak and minimum positions are reported per curve, but the first
    # minimum is shallow and noisy: one 0.02 A bin shifts coordination by
    # about 0.08. All coordination numbers therefore use the same common
    # cutoff as summarize_rdf_all_sizes.py.
    common_cutoff = args.cutoff
    rows = []
    for name in RDF_COLUMNS:
        g = rdf[name].to_numpy(dtype=float)
        features = first_shell_features(r, g, edges, density)
        rows.append({
            "rdf": name,
            "first_peak_r_A": features["first_peak_r_A"],
            "first_peak_g": features["first_peak_g"],
            "first_min_r_A": features["first_min_r_A"],
            "r_cut_A": common_cutoff,
            "coordination_number": coordination_to_cutoff(g, edges, density,
                                                          common_cutoff),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_dir / "first_shell.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for column, label, color in (
        ("delta_g_direct", "Direct: NNP − mW", "#343a40"),
        ("delta_g_NNP_to_mW", "NNP → mW − direct mW", "#2166ac"),
        ("delta_g_mW_to_NNP", "mW → NNP − direct NNP", "#d95f02"),
    ):
        ax.plot(r, residuals[column], color=color, lw=1.7, label=label)
    ax.axhline(0, color="#777777", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("RDF residual Δg(r)")
    ax.set_xlim(0, edges[-1])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.figure_prefix.with_suffix(f".{suffix}"), dpi=250)
    plt.close(fig)

    print(f"N={N}; box lengths [Å]: {lengths}; number density [Å^-3]: {density:.10f}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved {args.results_dir / 'first_shell.csv'}")
    print(f"Saved {args.results_dir / 'residuals.csv'}")
    print(f"Saved {args.figure_prefix}.png and .pdf")


if __name__ == "__main__":
    main()
