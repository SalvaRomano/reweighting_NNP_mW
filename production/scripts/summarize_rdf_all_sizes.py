#!/usr/bin/env python3
"""Summarize RDF first-shell coordination at one common cutoff for all sizes."""

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
from scripts.analyze_rdf_N256 import first_box_lengths  # noqa: E402
from src.rdf import radial_edges  # noqa: E402
from src.rdf_features import (COMMON_CUTOFF_A, coordination_to_cutoff,  # noqa: E402
                              first_shell_features)


SIZES = (64, 128, 256, 512, 1024, 2048)
DR = 0.02
COLORS = {"NNP": "#2166ac", "mW": "#d95f02"}
# Mass of one water molecule (18.015 u) in g, divided by 1 A^3 in cm^3.
WATER_MASS_G_CM3_PER_A3 = 18.015 * 1.66053906660


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdf-dir", type=Path, default=ROOT / "results/rdf_all_sizes")
    parser.add_argument("--rdf-N256", type=Path, default=ROOT / "analysis/rdf_N256.csv")
    parser.add_argument("--trajectory-root", type=Path, default=ROOT / "lammps")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/rdf_all_sizes/coordination_summary.csv")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/Structural")
    parser.add_argument("--cutoff", type=float, default=COMMON_CUTOFF_A,
                        help=f"Common coordination cutoff in A (default: {COMMON_CUTOFF_A})")
    return parser.parse_args()


def load_size(n, args):
    path = args.rdf_N256 if n == 256 else args.rdf_dir / f"rdf_N{n}.csv"
    rdf = pd.read_csv(path)
    columns = {"r", "g_NNP", "g_mW", "g_NNP_to_mW", "g_mW_to_NNP"}
    if not columns.issubset(rdf.columns):
        raise ValueError(f"Missing RDF columns in {path}: {columns - set(rdf.columns)}")
    if not np.isfinite(rdf[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite RDF data in {path}")

    mw_path = args.trajectory_root / f"mW_NVT/N_{n}/water.lammpstrj"
    nnp_path = args.trajectory_root / f"NN_NVT/N_{n}/run_1/water.lammpstrj"
    mw_lengths = first_box_lengths(mw_path, n)
    nnp_lengths = first_box_lengths(nnp_path, n)
    if not np.allclose(mw_lengths, nnp_lengths, rtol=0, atol=1e-6):
        raise ValueError(f"N={n}: NNP and mW box lengths differ")
    density = n / np.prod(mw_lengths)
    edges = radial_edges(DR, np.min(mw_lengths) / 2)
    r = rdf["r"].to_numpy(dtype=float)
    if len(r) != len(edges) - 1:
        raise ValueError(f"N={n}: RDF bin count does not match the trajectory box")

    mw_features = first_shell_features(r, rdf["g_mW"].to_numpy(), edges, density)
    nnp_features = first_shell_features(r, rdf["g_NNP"].to_numpy(), edges, density)
    cutoff = args.cutoff
    # One cutoff for all four curves and all sizes: coordination differences
    # then share the same radial range. The per-size mW minimum is kept only
    # as a diagnostic, because its position is set by bin-level noise.
    coords = {
        name: coordination_to_cutoff(rdf[name].to_numpy(dtype=float),
                                     edges, density, cutoff)
        for name in ("g_NNP", "g_mW", "g_NNP_to_mW", "g_mW_to_NNP")
    }
    return {
        "N": n,
        "r_peak_NNP": nnp_features["first_peak_r_A"],
        "r_peak_mW": mw_features["first_peak_r_A"],
        "peak_height_NNP": nnp_features["first_peak_g"],
        "peak_height_mW": mw_features["first_peak_g"],
        "number_density_A3": density,
        "r_min_mW": mw_features["first_min_r_A"],
        "r_cut": cutoff,
        "coord_NNP": coords["g_NNP"],
        "coord_mW": coords["g_mW"],
        "coord_NNP_to_mW": coords["g_NNP_to_mW"],
        "coord_mW_to_NNP": coords["g_mW_to_NNP"],
    }


def save_figure(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coordination(summary, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for column, label, color, linestyle, marker in (
        ("coord_NNP", "Direct NNP", COLORS["NNP"], "-", "o"),
        ("coord_mW", "Direct mW", COLORS["mW"], "-", "s"),
        ("coord_NNP_to_mW", "NNP → mW", COLORS["NNP"], "--", "^"),
        ("coord_mW_to_NNP", "mW → NNP", COLORS["mW"], "--", "v"),
    ):
        ax.plot(summary["N"], summary[column], marker=marker, ls=linestyle,
                color=color, lw=1.7, ms=6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    # Coordination at a fixed cutoff follows the NVT box density, which
    # differs between sizes, so show it under each size.
    mass_density = summary.set_index("N")["number_density_A3"] * WATER_MASS_G_CM3_PER_A3
    ax.set_xticklabels([f"{n}\n{mass_density[n]:.3f}" for n in SIZES])
    ax.set_xlabel(r"N  /  box density [g cm$^{-3}$]")
    ax.set_ylabel(rf"Coordination number ($r_c$ = {summary['r_cut'].iloc[0]:.2f} Å)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def plot_errors(summary, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    # The two before values are algebraically identical. Draw both with
    # different markers so the direction-specific labels remain visible.
    for column, label, color, linestyle, marker, markersize in (
        ("error_before_NNP_to_mW", "NNP → mW before", COLORS["NNP"], "-", "o", 8),
        ("error_after_NNP_to_mW", "NNP → mW after", COLORS["NNP"], "--", "^", 6),
        ("error_before_mW_to_NNP", "mW → NNP before", COLORS["mW"], ":", "s", 4),
        ("error_after_mW_to_NNP", "mW → NNP after", COLORS["mW"], "--", "v", 6),
    ):
        ax.plot(summary["N"], summary[column], marker=marker, ls=linestyle,
                color=color, lw=1.7, ms=markersize, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("N")
    ax.set_ylabel("Absolute coordination-number error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, path)


def main():
    args = parse_args()
    summary = pd.DataFrame([load_size(n, args) for n in SIZES])
    summary["error_before_NNP_to_mW"] = abs(summary.coord_NNP - summary.coord_mW)
    summary["error_after_NNP_to_mW"] = abs(summary.coord_NNP_to_mW - summary.coord_mW)
    summary["error_before_mW_to_NNP"] = abs(summary.coord_mW - summary.coord_NNP)
    summary["error_after_mW_to_NNP"] = abs(summary.coord_mW_to_NNP - summary.coord_NNP)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.12g")
    plot_coordination(summary, args.figure_dir / "rdf_coordination_all_sizes")
    plot_errors(summary, args.figure_dir / "rdf_coordination_errors_all_sizes")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved {args.output}")
    print(f"Saved summary figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
