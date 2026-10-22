#!/usr/bin/env python3
"""Bootstrap finite-sample bias of self-normalized reweighting."""

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
from src.ratio_bias import (bootstrap_ratio, population_ratio_moments,
                            weighted_inverse_n_fit)  # noqa: E402
import compute_energy_averages_and_reweighting as energy  # noqa: E402


KB_EV_PER_K = 8.617333262145e-5
N_BOOTSTRAP = 100_000
SEED = 20260923
FIT_CUTOFFS = (50, 100, 200, 500)
DIRECTIONS = (
    ("NNP -> mW", "NNP", (50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 20000)),
    ("mW -> NNP", "mW", (50, 100, 200, 500, 1000, 1500)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64,
                        help="System size N (default: 64)")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--fit-output", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/Bias")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Nominal NVT temperature in K (default: 300)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-fit-n", type=int, default=200,
                        help="Primary lower sample-size cutoff for the asymptotic fit")
    return parser.parse_args()


def analyze_direction(frame, system_size, direction, model, sample_sizes, temperature,
                      repetitions, rng):
    source = frame.loc[(frame["N"] == system_size) & (frame["model"] == model)].copy()
    if source.empty:
        raise ValueError(f"No N={system_size} source frames for {model}")
    for column in ("poteng", "cross_poteng"):
        source[column] = pd.to_numeric(source[column], errors="raise")
    if not np.isfinite(source[["poteng", "cross_poteng"]].to_numpy()).all():
        raise ValueError(f"Non-finite energies for {direction}")

    # The exponent uses the TOTAL cross-minus-source energy. Only the target
    # observable is divided by N to give energy per particle.
    delta_u_total = (source["cross_poteng"] - source["poteng"]).to_numpy(dtype=float)
    g = source["cross_poteng"].to_numpy(dtype=float) / system_size
    beta = 1.0 / (KB_EV_PER_K * temperature)
    logw = -beta * delta_u_total
    moments = population_ratio_moments(g, logw)

    # The one population-level shift inside `moments` defines these weights
    # once. Every bootstrap sample draws the same X and Y values.
    x = moments["x"]
    weights = moments["weights"]
    theta_full = moments["theta_full"]
    coefficient = moments["B_theory"]
    rows = []
    for n in sample_sizes:
        estimates = bootstrap_ratio(x, weights, n, repetitions, rng)
        mean_estimate = float(np.mean(estimates))
        std_estimate = float(np.std(estimates, ddof=1))
        sem = std_estimate / np.sqrt(repetitions)
        bias = mean_estimate - theta_full
        theory = coefficient / n
        rows.append({
            "direction": direction,
            "n": n,
            "n_bootstrap": repetitions,
            "theta_full": theta_full,
            "mean_theta_hat": mean_estimate,
            "std_theta_hat": std_estimate,
            "sem_theta_hat": sem,
            "bias_empirical": bias,
            "abs_bias_empirical": abs(bias),
            "relative_bias": bias / abs(theta_full),
            "B_theory": coefficient,
            "bias_theory": theory,
            "ratio_empirical_to_theory": bias / theory,
            "n_times_bias": n * bias,
        })
    summary = {
        "direction": direction,
        "n_source_frames": len(source),
        "theta_full": theta_full,
        "muX": moments["muX"],
        "muY": moments["muY"],
        "varY": moments["varY"],
        "covXY": moments["covXY"],
        "B_theory": coefficient,
        "neff_full": moments["neff_full"],
        "neff_fraction_full": moments["neff_full"] / len(source),
    }
    return rows, summary


def save_plots(results, system_size, direction, figure_dir):
    subset = results.loc[results["direction"] == direction].sort_values("n")
    stem = direction.replace(" ", "").replace("->", "_to_")

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.errorbar(subset["n"], subset["bias_empirical"],
                yerr=subset["sem_theta_hat"], marker="o", capsize=3,
                linestyle="none", color="#2166ac",
                label="Bootstrap empirical bias ± SEM")
    ax.plot(subset["n"], subset["bias_theory"], color="#d95f02", lw=1.8,
            label=r"Leading theory: $B/n$")
    ax.axhline(0, color="0.45", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Signed bias [eV/particle]")
    ax.set_title(direction)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figure_dir / f"bias_N{system_size}_{stem}{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.errorbar(subset["n"], subset["n_times_bias"],
                yerr=subset["n"] * subset["sem_theta_hat"], marker="o",
                linestyle="none", capsize=3, color="#2166ac",
                label=r"Bootstrap $n\,bias$")
    ax.axhline(subset["B_theory"].iloc[0], color="#d95f02", lw=1.8,
               label=r"$B_{theory}$")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$n \times bias_{empirical}$ [eV/particle]")
    ax.set_title(direction)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figure_dir / f"scaled_bias_N{system_size}_{stem}{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def fit_asymptotic_coefficients(results, cutoffs):
    """Fit B independently by direction for each requested lower cutoff."""
    rows = []
    for direction, _, _ in DIRECTIONS:
        direction_data = results.loc[results["direction"] == direction]
        theory = float(direction_data["B_theory"].iloc[0])
        for cutoff in cutoffs:
            fit_data = direction_data.loc[direction_data["n"] >= cutoff]
            if fit_data.empty:
                raise ValueError(f"No {direction} points at or above n={cutoff}")
            coefficient, coefficient_sem = weighted_inverse_n_fit(
                fit_data["n"], fit_data["bias_empirical"], fit_data["sem_theta_hat"])
            rows.append({
                "direction": direction,
                "n_min_fit": cutoff,
                "B_empirical": coefficient,
                "B_empirical_sem": coefficient_sem,
                "B_theory": theory,
                "B_empirical_over_B_theory": coefficient / theory,
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    output = args.output or ROOT / f"analysis/bias_N{args.size}.csv"
    summary_output = args.summary or ROOT / f"analysis/bias_N{args.size}_summary.csv"
    fit_output = args.fit_output or ROOT / f"analysis/bias_N{args.size}_fits.csv"
    if args.temperature <= 0 or args.n_bootstrap < 100_000:
        raise ValueError("Temperature must be positive and at least 100,000 bootstraps are required")
    if args.min_fit_n <= 0:
        raise ValueError("--min-fit-n must be positive")
    frame = pd.read_csv(args.input)
    required = {"model", "N", "source_run", "step", "poteng", "cross_poteng"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(frame.columns))}")
    # Use the same endpoint convention as the reweighting and cumulant
    # analyses: drop the first and last saved frame of every original
    # trajectory. The first frame is the shared, unequilibrated starting
    # configuration of all NNP runs, so keeping it adds repeated outliers.
    frame, removal = energy.remove_trajectory_endpoints(frame)
    print(f"Removed {int(removal['n_removed'].sum())} endpoint frames from "
          f"{len(removal)} trajectories")

    seed_sequences = np.random.SeedSequence(args.seed).spawn(len(DIRECTIONS))
    all_rows, summaries = [], []
    for definition, seed_sequence in zip(DIRECTIONS, seed_sequences):
        direction, model, sample_sizes = definition
        rows, summary = analyze_direction(
            frame, args.size, direction, model, sample_sizes, args.temperature,
            args.n_bootstrap, np.random.default_rng(seed_sequence))
        all_rows.extend(rows)
        summaries.append(summary)

    results = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(summaries)
    cutoffs = tuple(dict.fromkeys((*FIT_CUTOFFS, args.min_fit_n)))
    fits = fit_asymptotic_coefficients(results, cutoffs)
    output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, float_format="%.15g")
    summary_frame.to_csv(summary_output, index=False, float_format="%.15g")
    fits.to_csv(fit_output, index=False, float_format="%.15g")
    for direction, _, _ in DIRECTIONS:
        save_plots(results, args.size, direction, args.figure_dir)

    print("Full-ensemble quantities:")
    print(summary_frame.to_string(index=False, float_format=lambda x: f"{x:.9g}"))
    print("\nEmpirical versus theoretical bias:")
    print(results[["direction", "n", "bias_empirical", "sem_theta_hat",
                   "bias_theory", "ratio_empirical_to_theory", "n_times_bias"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nAsymptotic weighted fits, bias = B_empirical/n (no intercept):")
    print(fits.to_string(index=False, float_format=lambda x: f"{x:.9g}"))
    primary = fits.loc[fits["n_min_fit"] == args.min_fit_n]
    print(f"\nPrimary fit cutoff: n >= {args.min_fit_n}")
    print(primary.to_string(index=False, float_format=lambda x: f"{x:.9g}"))
    print(f"\nSaved {output}, {summary_output}, and {fit_output}")


if __name__ == "__main__":
    main()
