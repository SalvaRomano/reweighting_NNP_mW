#!/usr/bin/env python3
"""Combine both reweighting-direction bias curves on common n values."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "NNP -> mW": ("#2166ac", "o", "NNP → mW"),
    "mW -> NNP": ("#d95f02", "s", "mW → NNP"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--scaled-output-prefix", type=Path)
    parser.add_argument("--band-summary", type=Path,
                        default=ROOT / "results/bias_vs_size/summary.csv",
                        help="B_theory 95%% intervals from analyze_bias_vs_size.py")
    parser.add_argument("--min-n", type=int, default=100,
                        help="Smallest n in the scaled figure (default: 100)")
    parser.add_argument("--skip-scaled", action="store_true",
                        help="Do not generate the separate n-times-bias diagnostic")
    return parser.parse_args()


def plot_scaled(selected, size, min_n, band_summary, output_prefix):
    """n*bias against the leading-order coefficient B_theory and its interval.

    Only the leading 1/n term is compared, so small n (where the next order
    still matters at large N) are left out. The band is the frame-resampling
    95% interval of B_theory: how well the finite dataset determines B.
    """
    bands = None
    if band_summary.is_file():
        bands = pd.read_csv(band_summary)
        bands = bands.loc[bands["N"] == size].set_index("direction")
    data = selected.loc[selected["n"] >= min_n]
    if data.empty:
        raise ValueError(f"No common sample sizes at or above n={min_n}")
    sizes = sorted(data["n"].unique())

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for direction, (color, marker, label) in STYLES.items():
        subset = data.loc[data["direction"] == direction].sort_values("n")
        theory = float(subset["B_theory"].iloc[0])
        if bands is not None and direction in bands.index:
            ax.axhspan(bands.loc[direction, "B_ci95_low"], bands.loc[direction, "B_ci95_high"],
                       color=color, alpha=0.12, lw=0,
                       label=f"{label}: 95% interval of $B$ from the data")
        ax.axhline(theory, color=color, lw=1.8, label=rf"{label}: $B_{{theory}}$")
        ax.errorbar(subset["n"], subset["n"] * subset["bias_empirical"],
                    yerr=subset["n"] * subset["sem_theta_hat"], marker=marker,
                    linestyle="none", color=color, capsize=3, ms=7,
                    label=rf"{label}: bootstrap $n\,$bias ± $n\,$SEM")
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"$n \times$ bias [eV/particle]")
    ax.set_title(f"Leading-order bias coefficient, N = {size}")
    ax.grid(alpha=0.25)
    # Legend below the axes: the bands fill most of the plotting area.
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    args = parse_args()
    input_path = args.input or ROOT / f"analysis/bias_N{args.size}.csv"
    output_prefix = args.output_prefix or ROOT / f"figures/Bias/bias_N{args.size}_combined"
    results = pd.read_csv(input_path)
    required = {"direction", "n", "bias_empirical", "sem_theta_hat", "B_theory"}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(results.columns))}")

    # Restrict both directions to exactly the intersection of their sampled n.
    sample_sets = [set(group["n"]) for _, group in results.groupby("direction")]
    common_n = sorted(set.intersection(*sample_sets))
    if not common_n:
        raise ValueError("The directions have no common sample sizes")
    selected = results.loc[results["n"].isin(common_n)].copy()
    for direction in STYLES:
        observed = selected.loc[selected["direction"] == direction, "n"].tolist()
        if sorted(observed) != common_n:
            raise ValueError(f"Incomplete common sample-size coverage for {direction}")

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for direction, (color, marker, label) in STYLES.items():
        subset = selected.loc[selected["direction"] == direction].sort_values("n")
        ax.errorbar(subset["n"], subset["bias_empirical"],
                    yerr=subset["sem_theta_hat"], marker=marker,
                    linestyle="none", color=color, capsize=2, markersize=4.2)
        ax.plot(subset["n"], subset["B_theory"] / subset["n"],
                color=color, linewidth=1.35)
    ax.axhline(0, color="0.45", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xticks(common_n)
    ax.set_xticklabels([str(n) for n in common_n], rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_xlabel("Sample size, n", fontsize=9)
    ax.set_ylabel("Bias (eV/particle)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color=STYLES["NNP -> mW"][0], linewidth=1.35,
               label="NNP → mW: Theory"),
        Line2D([0], [0], color=STYLES["mW -> NNP"][0], linewidth=1.35,
               label="mW → NNP: Theory"),
        Line2D([0], [0], color=STYLES["NNP -> mW"][0], marker="o",
               linestyle="none", markersize=4.2, label="NNP → mW: Bootstrap"),
        Line2D([0], [0], color=STYLES["mW -> NNP"][0], marker="s",
               linestyle="none", markersize=4.2, label="mW → NNP: Bootstrap"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6.0, ncol=2,
              loc="upper right", handlelength=1.3, columnspacing=0.5,
              borderaxespad=0.25)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)
    print(f"Common sample sizes: {common_n}")
    if not args.skip_scaled:
        scaled_prefix = (args.scaled_output_prefix
                         or ROOT / f"figures/Bias/scaled_bias_N{args.size}_combined")
        plot_scaled(selected, args.size, args.min_n, args.band_summary, scaled_prefix)


if __name__ == "__main__":
    main()
