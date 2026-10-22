#!/usr/bin/env python3
"""Leading-order (1/n) ratio-estimator bias versus system size.

For every N and reweighting direction this computes, from the full source
dataset after endpoint removal:

- B_theory, the 1/n bias coefficient, with a 95% interval from resampling the
  source frames (how well the finite dataset determines B);
- its two factors, B = -(M/N_eff) * (theta_w2 - theta);
- |bias|/SE at the actual sample size n = M, with SE = sigma/sqrt(M) the
  leading-order standard error of the ratio estimator.

Only the leading 1/n term is used. It is reliable when n >> M/N_eff, so the
ratio n/(M/N_eff) = N_eff is reported as the validity measure.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ratio_bias import leading_order_terms, population_ratio_moments  # noqa: E402
import compute_energy_averages_and_reweighting as energy  # noqa: E402


SEED = 20261006
DIRECTIONS = (("NNP -> mW", "NNP"), ("mW -> NNP", "mW"))
STYLES = {
    "NNP -> mW": ("#2166ac", "o", "NNP → mW"),
    "mW -> NNP": ("#d95f02", "s", "mW → NNP"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/bias_vs_size")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/Bias/vs_size")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--n-resample", type=int, default=2000,
                        help="Frame resamples for the B_theory interval (default: 2000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-neff", type=float, default=100.0,
                        help="Below this N_eff = n/(M/N_eff) the 1/n expansion is "
                        "flagged as marginal (default: 100)")
    return parser.parse_args()


def analyze(frame, temperature, n_resample, rng):
    beta = 1.0 / (energy.KB_EV_PER_K * temperature)
    rows = []
    for size in sorted(frame["N"].unique()):
        for direction, model in DIRECTIONS:
            source = frame.loc[(frame["N"] == size) & (frame["model"] == model)]
            if source.empty:
                raise ValueError(f"No N={size} source frames for {model}")
            # Total energy difference in the exponent; only G is per particle.
            g = (source["cross_poteng"] / size).to_numpy(dtype=float)
            logw = (-beta * (source["cross_poteng"] - source["poteng"])).to_numpy(dtype=float)
            terms = leading_order_terms(g, logw)

            # Resample the source frames to see how well this finite dataset
            # determines B. This is separate from the n-sample bootstrap of
            # analyze_ratio_bias.py, which treats the dataset as the population.
            count = len(g)
            resampled = np.empty(n_resample)
            for i in range(n_resample):
                index = rng.integers(0, count, count)
                resampled[i] = population_ratio_moments(g[index], logw[index])["B_theory"]
            low, high = np.percentile(resampled, [2.5, 97.5])

            # Leading-order bias and standard error at the full sample size.
            bias_at_m = terms["B_theory"] / count
            se_at_m = terms["sigma"] / np.sqrt(count)
            rows.append({
                "N": int(size),
                "direction": direction,
                "n_source_frames": count,
                "theta_full": terms["theta_full"],
                "B_theory": terms["B_theory"],
                "B_ci95_low": low,
                "B_ci95_high": high,
                "B_ci_includes_zero": bool(low <= 0 <= high),
                "neff_full": terms["neff_full"],
                "inefficiency_M_over_neff": terms["inefficiency"],
                "theta_w2": terms["theta_w2"],
                "delta_theta_w2": terms["delta_theta_w2"],
                "sigma": terms["sigma"],
                "bias_at_M": bias_at_m,
                "se_at_M": se_at_m,
                "bias_over_se_at_M": abs(bias_at_m) / se_at_m,
            })
    return pd.DataFrame(rows)


def setup_size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("N")
    ax.grid(alpha=0.25)


def save_figure(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient(summary, path):
    """|B_theory| with its frame-resampling 95% interval, log scale."""
    sizes = sorted(summary["N"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    floor = 0.2 * summary["B_theory"].abs().min()
    offsets = {"NNP -> mW": 0.97, "mW -> NNP": 1.03}  # avoid overlapping bars
    for direction, (color, marker, label) in STYLES.items():
        data = summary.loc[summary["direction"] == direction].sort_values("N")
        x = data["N"] * offsets[direction]
        value = data["B_theory"].abs()
        # Interval of |B|: when the signed interval contains zero, the lower
        # end reaches the axis floor and the marker is drawn open.
        ends = data[["B_ci95_low", "B_ci95_high"]].abs()
        low = np.where(data["B_ci_includes_zero"], floor, ends.min(axis=1))
        high = ends.max(axis=1)
        ax.plot(x, value, color=color, lw=1.4, alpha=0.6)
        ax.errorbar(x, value, yerr=[value - low, high - value], fmt="none",
                    ecolor=color, capsize=3, lw=1.2)
        for xi, vi, open_marker in zip(x, value, data["B_ci_includes_zero"]):
            ax.plot(xi, vi, marker=marker, ms=7, color=color,
                    mfc="white" if open_marker else color, mew=1.5)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    setup_size_axis(ax, sizes)
    ax.set_ylabel(r"$|B_{theory}|$ [eV/particle]   (bias $= B/n$)")
    handles = [Line2D([], [], color=c, marker=m, ms=7, label=l) for c, m, l in STYLES.values()]
    handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls="none", ms=7,
                          label="95% interval includes zero"))
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def plot_decomposition(summary, path):
    """Two panels: the factors of B = -(M/N_eff) (theta_w2 - theta)."""
    sizes = sorted(summary["N"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    panels = (
        ("inefficiency_M_over_neff", r"Weight inefficiency $M/N_{eff}$"),
        ("delta_theta_w2", r"$|\theta_{w^2} - \theta|$ [eV/particle]"),
    )
    for ax, (column, ylabel) in zip(axes, panels):
        for direction, (color, marker, label) in STYLES.items():
            data = summary.loc[summary["direction"] == direction].sort_values("N")
            ax.plot(data["N"], data[column].abs(), color=color, marker=marker,
                    ms=7, lw=1.7, label=label)
        ax.set_yscale("log")
        setup_size_axis(ax, sizes)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Overlap loss", fontsize=11)
    axes[1].set_title("Correlation of G with the weights", fontsize=11)
    fig.suptitle(r"$B_{theory} = -(M/N_{eff})\,(\theta_{w^2} - \theta)$", fontsize=12)
    fig.tight_layout()
    save_figure(fig, path)


def plot_bias_to_noise(summary, path, min_neff):
    """Leading-order |bias|/SE at the actual sample size n = M."""
    sizes = sorted(summary["N"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for direction, (color, marker, label) in STYLES.items():
        data = summary.loc[summary["direction"] == direction].sort_values("N")
        ax.plot(data["N"], data["bias_over_se_at_M"], color=color, lw=1.7, label=label)
        for _, row in data.iterrows():
            # N_eff = n/(M/N_eff) at n = M: few effective samples means the
            # 1/n expansion itself is only marginally valid.
            marginal = row["neff_full"] < min_neff
            ax.plot(row["N"], row["bias_over_se_at_M"], marker=marker, ms=7,
                    color=color, mfc="white" if marginal else color, mew=1.5)
    ax.axhline(1.0, color="0.35", lw=1.0, ls="--")
    ax.text(sizes[0], 1.25, "bias = statistical error", color="0.35", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(top=4.0)
    setup_size_axis(ax, sizes)
    ax.set_ylabel(r"$|B/M|\,/\,(\sigma/\sqrt{M})$ at the full sample size $M$")
    handles = [Line2D([], [], color=c, marker=m, ms=7, label=l) for c, m, l in STYLES.values()]
    handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls="none", ms=7,
                          label=rf"$N_{{eff}} < {min_neff:g}$: 1/n expansion marginal"))
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, path)


def main():
    args = parse_args()
    if args.temperature <= 0 or args.n_resample < 100:
        raise ValueError("Temperature must be positive and --n-resample at least 100")
    frame = pd.read_csv(args.input)
    required = {"model", "N", "source_run", "step", "poteng", "cross_poteng"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(frame.columns))}")
    # Same endpoint convention as every other analysis.
    frame, removal = energy.remove_trajectory_endpoints(frame)
    print(f"Removed {int(removal['n_removed'].sum())} endpoint frames from "
          f"{len(removal)} trajectories")

    summary = analyze(frame, args.temperature, args.n_resample,
                      np.random.default_rng(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.csv"
    summary.to_csv(output, index=False, float_format="%.12g")
    plot_coefficient(summary, args.figure_dir / "bias_coefficient_vs_N")
    plot_decomposition(summary, args.figure_dir / "bias_decomposition_vs_N")
    plot_bias_to_noise(summary, args.figure_dir / "bias_to_noise_vs_N", args.min_neff)

    columns = ["N", "direction", "n_source_frames", "B_theory", "B_ci95_low",
               "B_ci95_high", "inefficiency_M_over_neff", "delta_theta_w2",
               "bias_over_se_at_M", "neff_full"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"Saved {output} and figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
