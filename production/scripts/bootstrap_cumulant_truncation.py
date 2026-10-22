#!/usr/bin/env python3
"""Bootstrap truncation residuals without changing existing production outputs."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cumulant_expansion import prepare_observables, contributions
from src.cumulant_truncation import (signed_truncation_residual, bootstrap_residuals,
                                     summarize_residuals)
import compute_energy_averages_and_reweighting as energy

DIRECTIONS = (("NNP", "mW", "o"), ("mW", "NNP", "s"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulant_truncation_bootstrap.csv")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/Cumulant/appendix")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    return parser.parse_args()


def validate_small_subset(V, A):
    """Check paired draws on four actual N=64 frames with explicit low-order formulas."""
    V, A = V[:4], A[:4]
    seed = 13
    _, draws = bootstrap_residuals(V, A, 3, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    for b in range(3):
        index = rng.integers(0, 4, 4)
        v, a = V[index], A[index]
        # Independently calculate the full ratio and explicit mixed cumulants.
        weights = np.exp(v - v.max())
        full = np.sum(a * weights) / np.sum(weights)
        explicit = contributions(v, a)
        expected = np.array([explicit[f"G_order{k}"] - full for k in range(5)])
        np.testing.assert_allclose(draws[b, :5], expected, rtol=0, atol=2e-16)
    print("Validated three paired bootstrap draws on four retained N=64 frames.", flush=True)


def plot_appendix(summary, figure_dir):
    # Preserve the all-size truncation plot's tab10 colors (1024=purple, 2048=brown).
    sizes = sorted(summary.N.unique())
    colors = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(sizes)}
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "mathtext.fontset": "cm",
                         "pdf.fonttype": 42, "axes.linewidth": 0.9}):
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.8), sharex=True,
                                 layout="constrained")
        floor = 1e-16
        for ax, n in zip(axes, (1024, 2048)):
            for source, target, marker in DIRECTIONS:
                data = summary.loc[(summary.N == n)
                                   & (summary.direction == f"{source}_to_{target}")]
                data = data.sort_values("order")
                x = data.order.to_numpy()
                # Clip only displayed values, never the saved numerical summaries.
                for low, high, alpha in (("p2_5", "p97_5", 0.08), ("p16", "p84", 0.20)):
                    ax.fill_between(x, np.maximum(data[f"bootstrap_epsilon_{low}"], floor),
                                    np.maximum(data[f"bootstrap_epsilon_{high}"], floor),
                                    color=colors[n], alpha=alpha, linewidth=0)
                ax.plot(x, np.maximum(data.original_epsilon, floor), color=colors[n],
                        marker=marker, ms=3.2, lw=1, label=f"{source} → {target}")
            ax.set_yscale("log")
            ax.text(0.97, 0.95, rf"$N={n}$", transform=ax.transAxes,
                    ha="right", va="top")
            ax.set_ylabel(r"$\epsilon_m^{\mathrm{tr}}$ (eV/particle)")
            ax.grid(alpha=0.25)
            ax.tick_params(which="major", length=4, width=0.9)
        axes[-1].set_xticks(np.arange(0, 13, 2))
        axes[-1].set_xticks(np.arange(1, 12, 2), minor=True)
        axes[-1].set_xlabel(r"Expansion order $m$")
        handles = [Line2D([], [], color="0.3", marker=marker, ms=3.5, ls="none",
                          label=f"{source} → {target}") for source, target, marker in DIRECTIONS]
        axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=6.8)
        figure_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            path = figure_dir / f"truncation_error_bootstrap_large_N.{suffix}"
            fig.savefig(path, dpi=600)
            print(f"Saved {path}")
        plt.close(fig)


def main():
    args = parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("At least two bootstrap realizations are required")
    protected = {args.input.resolve(), args.reference.resolve(),
                 (ROOT / "results/cumulant_expansion/summary.csv").resolve()}
    if args.output.resolve() in protected:
        raise ValueError("Output must not overwrite input or production results")
    frame = energy.load_and_validate(args.input)
    trimmed, removal = energy.remove_trajectory_endpoints(frame)
    reference = pd.read_csv(args.reference, float_precision="round_trip")
    reference = reference.set_index(["N", "direction", "order"])
    if reference.index.has_duplicates:
        raise ValueError("Duplicate reference rows")
    beta = 1 / (energy.KB_EV_PER_K * 300.0)
    groups = []
    # Validate every original residual before starting any bootstrap work.
    for n in sorted(trimmed.N.unique()):
        for source, target, _ in DIRECTIONS:
            group = trimmed.loc[(trimmed.N == n) & (trimmed.model == source)]
            V, A = prepare_observables(group.poteng, group.cross_poteng, int(n), beta)
            direction = f"{source}_to_{target}"
            original = signed_truncation_residual(V, A)
            expected = reference.loc[[(n, direction, m) for m in range(13)],
                                     "error_vs_full_reweighting"].to_numpy()
            # Round-trip CSV loading allows bit-for-bit reproduction, even for tiny residuals.
            np.testing.assert_array_equal(original, expected)
            groups.append((int(n), direction, V, A))
    print(f"Exactly reproduced all {len(groups) * 13} original residuals; "
          f"removed {removal.n_removed.sum()} endpoint frames.", flush=True)
    validate_small_subset(groups[0][2], groups[0][3])
    rng = np.random.default_rng(args.seed)
    summaries = []
    for n, direction, V, A in groups:
        original, draws = bootstrap_residuals(V, A, args.n_bootstrap, rng)
        table = pd.DataFrame(summarize_residuals(original, draws))
        table.insert(0, "order", np.arange(13))
        table.insert(0, "direction", direction)
        table.insert(0, "N", n)
        table["n_source_frames"] = len(V)
        table["n_bootstrap"] = args.n_bootstrap
        table["seed"] = args.seed
        table["temperature_K"] = 300.0
        summaries.append(table)
        print(f"Completed N={n} {direction}: M={len(V)}, B={args.n_bootstrap}", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.17g")
    print(f"Saved {args.output}")
    selected = summary.loc[summary.N.isin([1024, 2048]) & summary.order.isin([4, 8, 12])]
    columns = ["N", "direction", "order", "original_epsilon", "bootstrap_median_epsilon",
               "bootstrap_epsilon_p16", "bootstrap_epsilon_p84",
               "bootstrap_epsilon_p2_5", "bootstrap_epsilon_p97_5"]
    print("\nTruncation residuals and bootstrap intervals [eV/particle]:")
    print(selected[columns].to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    plot_appendix(summary, args.figure_dir)


if __name__ == "__main__":
    main()
