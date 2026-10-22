#!/usr/bin/env python3
"""Fit exponential system-size decay of the relative effective sample size."""

from __future__ import annotations

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
DIRECTIONS = (
    ("NNP -> mW", "neff_fraction_NNP_to_mW", "tab:blue", "s"),
    ("mW -> NNP", "neff_fraction_mW_to_NNP", "tab:orange", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "analysis/energy_averages_and_reweighting.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures/Reweighting",
    )
    return parser.parse_args()


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    """Fit y = intercept + slope*x by unweighted ordinary least squares."""
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "A": float(np.exp(intercept)),
        "c": float(-slope),
        "R2": r_squared,
        "fitted": fitted,
        "residuals": residuals,
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str,
                fixed_canvas: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_options = {} if fixed_canvas else {"bbox_inches": "tight"}
    fig.savefig(output_dir / f"{stem}.png", dpi=300, **save_options)
    fig.savefig(output_dir / f"{stem}.pdf", **save_options)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"N", *(column for _, column, _, _ in DIRECTIONS)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame.sort_values("N", kind="stable").reset_index(drop=True)
    sizes = frame["N"].to_numpy(dtype=float)
    if not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("System sizes must be finite and positive")

    fits = {}
    parameter_rows = []
    for direction, column, _, _ in DIRECTIONS:
        fractions = frame[column].to_numpy(dtype=float)
        if not np.isfinite(fractions).all() or (fractions <= 0).any():
            raise ValueError(f"{column} must contain finite, positive relative ESS values")
        fits[direction] = linear_fit(sizes, np.log(fractions))
        fit = fits[direction]
        parameter_rows.append({
            "direction": direction,
            "A": fit["A"],
            "c": fit["c"],
            "R2": fit["R2"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = pd.DataFrame(parameter_rows)
    parameter_path = args.output_dir / "ess_exponential_fit_parameters.csv"
    parameters.to_csv(parameter_path, index=False, float_format="%.12g")

    line_sizes = np.linspace(float(sizes.min()), float(sizes.max()), 500)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for direction, column, color, marker in DIRECTIONS:
        fit = fits[direction]
        values = np.log(frame[column].to_numpy(dtype=float))
        ax.scatter(sizes, values, s=22, marker=marker, color=color, zorder=3)
        ax.plot(line_sizes, fit["intercept"] + fit["slope"] * line_sizes,
                color=color, linewidth=1.45)
    ax.set_xlabel("System size, N", fontsize=9)
    ax.set_ylabel(r"$\ln(n_{\mathrm{eff}}/n)$", fontsize=9)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(size)}" for size in sizes], rotation=90,
                       ha="center", va="top")
    ax.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linewidth=1.6, label="NNP → mW"),
        Line2D([0], [0], color="tab:orange", linewidth=1.6, label="mW → NNP"),
        Line2D([0], [0], color="0.25", linewidth=1.5, label="Fit"),
        Line2D([0], [0], color="0.25", marker="s", linestyle="None",
               markersize=4.5, label="Data"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6.8, ncol=2,
              handlelength=2.1, columnspacing=0.8)
    save_figure(fig, args.output_dir, "ess_log_vs_N_exponential_fit",
                fixed_canvas=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for direction, _, color, marker in DIRECTIONS:
        fit = fits[direction]
        ax.plot(sizes, fit["residuals"], marker=marker, markersize=7,
                linewidth=2.0, color=color, label=direction.replace("->", "→"))
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.2)
    ax.set_xlabel("System size, N", fontsize=16)
    ax.set_ylabel(r"Residual in $\ln(n_{\mathrm{eff}}/n)$", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=12)
    save_figure(fig, args.output_dir, "ess_exponential_fit_residuals")

    print("Exponential relative-ESS fits: f_ESS(N) = A exp(-c N)")
    print(parameters.to_string(index=False, float_format=lambda value: f"{value:.12g}"))
    print(f"Saved {parameter_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
