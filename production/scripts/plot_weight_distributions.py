#!/usr/bin/env python3
"""Plot stabilized reweighting-weight distributions for every system size."""

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
sys.path.insert(0, str(ROOT / "scripts"))
import compute_energy_averages_and_reweighting as energy  # noqa: E402

KB_EV_PER_K = 8.617333262145e-5
SIZES = (64, 128, 256, 512, 1024, 2048)
DIRECTIONS = (
    ("NNP", "NNP → mW", "tab:blue"),
    ("mW", "mW → NNP", "tab:orange"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "analysis/thermo_all_with_cross_merged.csv",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--output-prefix", type=Path,
        default=ROOT / "figures/Reweighting/weight_distributions_all_sizes",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "results/weight_distributions_all_sizes_summary.csv",
    )
    return parser.parse_args()


def relative_log10_weights(frame: pd.DataFrame, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return log10(n*p_i) and normalized probabilities using total Delta U."""
    logw = -beta * (frame["cross_poteng"] - frame["poteng"]).to_numpy(dtype=float)
    shift = float(np.max(logw))
    shifted = np.exp(logw - shift)
    probabilities = shifted / shifted.sum()
    # n*p_i = w_i/<w> is comparable across datasets with different counts.
    log10_relative = (logw - shift - np.log(np.mean(shifted))) / np.log(10.0)
    return log10_relative, probabilities


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    frame = pd.read_csv(args.input)
    frame, removal = energy.remove_trajectory_endpoints(frame)
    beta = 1.0 / (KB_EV_PER_K * args.temperature)

    distributions = {}
    rows = []
    for size in SIZES:
        for model, direction, _ in DIRECTIONS:
            subset = frame.loc[(frame["N"] == size) & (frame["model"] == model)]
            if subset.empty:
                raise ValueError(f"No frames for N={size}, model={model}")
            values, probabilities = relative_log10_weights(subset, beta)
            distributions[(size, model)] = values
            quantiles = np.quantile(values, [0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0])
            ess = 1.0 / np.sum(probabilities**2)
            rows.append({
                "N": size,
                "direction": direction,
                "n_frames": len(values),
                "log10_npi_min": quantiles[0],
                "log10_npi_q01": quantiles[1],
                "log10_npi_q10": quantiles[2],
                "log10_npi_median": quantiles[3],
                "log10_npi_q90": quantiles[4],
                "log10_npi_q99": quantiles[5],
                "log10_npi_max": quantiles[6],
                "max_probability": float(probabilities.max()),
                "ESS": ess,
                "relative_ESS": ess / len(values),
            })

    # A common binning range makes broadening with N directly comparable.
    all_values = np.concatenate(list(distributions.values()))
    lower = np.floor(all_values.min() * 2.0) / 2.0
    upper = np.ceil(all_values.max() * 2.0) / 2.0
    bins = np.linspace(lower, upper, 91)
    # Symmetric limits retain the entire distribution in both directions.
    x_extent = max(4.0, 2.0 * np.ceil(max(abs(lower), abs(upper)) / 2.0))

    # Two-column manuscript width; shared labels keep the six narrow panels readable.
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "mathtext.fontset": "cm", "pdf.fonttype": 42,
                         "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 2.25), sharex=True, sharey=True,
                             layout="constrained")
    for ax, size in zip(axes, SIZES):
        for model, direction, color in DIRECTIONS:
            ax.hist(distributions[(size, model)], bins=bins, density=True,
                    histtype="step", linewidth=1.1, color=color, label=direction)
        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.7)
        ax.set_yscale("log")
        ax.text(0.05, 0.95, rf"$N={size}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=9)
        ax.grid(alpha=0.22)
        ax.tick_params(which="major", length=3, width=0.8)
        ax.tick_params(which="minor", length=1.5, width=0.6)
        ax.set_xlim(-x_extent, x_extent)
        ax.set_xticks([-4, -2, 0, 2, 4])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2,
               frameon=False, fontsize=8)
    fig.supxlabel(r"$\log_{10}(n p_i)$", fontsize=11)
    axes[0].set_ylabel("Probability density", fontsize=9)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.summary, index=False, float_format="%.12g")
    print(f"Removed {int(removal['n_removed'].sum())} endpoint frames")
    print(f"Saved {args.output_prefix.with_suffix('.*')} and {args.summary}")


if __name__ == "__main__":
    main()
