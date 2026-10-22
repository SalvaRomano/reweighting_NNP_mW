#!/usr/bin/env python3
"""Plot corrected NNP-to-mW diagnostics and per-trajectory thermodynamic traces."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_energy_averages_and_reweighting import load_and_validate, remove_trajectory_endpoints
from plot_reweighting_diagnostics import save_figure, style_axis
from plot_thermo_vs_configuration import QUANTITIES, plot_system_size


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "results/corrected_mw_rerun")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "figures/Reweighting/corrected_mw_rerun")
    args = parser.parse_args()
    summary = pd.read_csv(args.input_dir / "summary_by_size.csv").sort_values("N")
    metrics = [
        ("neff_fraction", "Effective sample size fraction"),
        ("max_weight_fraction", "Largest normalized weight fraction"),
        ("top10_weight_fraction", "Top-10 normalized weight fraction"),
    ]
    if summary.empty or summary.N.duplicated().any() or (summary.N <= 0).any():
        raise ValueError("Expected one summary row per positive system size")
    for column, _ in metrics:
        values = summary[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Invalid fractions: {column}")
    if (summary.neff_fraction <= 0).any():
        raise ValueError("ESS fractions must be positive for the logarithmic axis")

    frame = load_and_validate(args.input_dir / "nnp_frames_with_corrected_mw.csv")
    if set(frame.model) != {"NNP"}:
        raise ValueError("Expected only the corrected NNP source ensemble")
    frame, removal = remove_trajectory_endpoints(frame)
    counts = frame.groupby("N").size()
    if not counts.equals(summary.set_index("N").n_frames.astype("int64")):
        raise ValueError("Trace frame counts disagree with the reweighting summary")

    # Restore original run IDs for concatenation and boundary markers. Trimming
    # uses source_run so each trajectory loses exactly its own two endpoints.
    frame["run"] = frame.source_run
    frame["poteng_per_particle"] = frame.poteng / frame.N
    frame["cross_poteng_per_particle"] = frame.cross_poteng / frame.N
    frame["delta_u"] = frame.delta_U / frame.N
    if "toteng" in frame:
        frame["toteng_per_particle"] = frame.toteng / frame.N
    quantities = [(column, label) for column, label in QUANTITIES
                  if column in frame and frame[column].notna().any()]

    for column, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(summary.N, summary[column], marker="o", linewidth=1.8,
                markersize=5, color="tab:blue", label="NNP → mW")
        if column == "neff_fraction":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)
        style_axis(ax, summary.N.to_numpy(dtype=float), ylabel)
        save_figure(fig, args.output_dir, column + "_vs_N")
    for size in sorted(frame.N.unique()):
        plot_system_size(frame, int(size), quantities,
                         args.output_dir / "thermo_vs_configuration")
    print(f"Plotted {len(frame):,} frames after removing {removal.n_removed.sum()} endpoints.")
    print("Reweighting-error plot unavailable: no direct corrected-mW ensemble.")


if __name__ == "__main__":
    main()
