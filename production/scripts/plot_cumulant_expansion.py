#!/usr/bin/env python3
"""Plot signed cumulant coefficients with bootstrap errors and series convergence."""

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
DIRECTIONS = (("NNP_to_mW", "NNP → mW"), ("mW_to_NNP", "mW → NNP"))
MARKERS = ("o", "s", "^", "D", "v", "P")
REQUIRED_COLUMNS = {"N", "direction", "temperature_K", "G_full_reweighted"}
REQUIRED_COLUMNS.update(f"G_order{order}" for order in range(5))
REQUIRED_COLUMNS.update(f"C{order}" for order in range(1, 5))
REQUIRED_COLUMNS.update(f"C{order}_se" for order in range(1, 5))
REQUIRED_COLUMNS.update(f"C{order}_abs_over_se" for order in range(1, 5))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulant_expansion/summary.csv")
    parser.add_argument("--series-input", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/Cumulant")
    parser.add_argument("--resolved-threshold", type=float, default=3.0,
                        help="|C_k|/SE at or above which a coefficient counts as "
                        "resolved (default: 3)")
    return parser.parse_args()


def load_summary(path):
    """Validate one finite, positive-size row per direction and system size."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    numeric_columns = REQUIRED_COLUMNS - {"direction"}
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values")
    if (frame["N"] <= 0).any() or not np.equal(frame["N"], np.floor(frame["N"])).all():
        raise ValueError("N must contain positive integers")
    if (frame["temperature_K"] <= 0).any() or frame["temperature_K"].nunique() != 1:
        raise ValueError("The summary must contain one positive nominal temperature")
    if frame.duplicated(["N", "direction"]).any():
        raise ValueError("The summary contains duplicate N/direction rows")
    expected_directions = {name for name, _ in DIRECTIONS}
    if set(frame["direction"]) != expected_directions:
        raise ValueError("The summary must contain both reweighting directions")
    for direction, _ in DIRECTIONS:
        sizes = set(frame.loc[frame["direction"] == direction, "N"])
        if sizes != set(frame["N"]):
            raise ValueError(f"Missing system size in direction {direction}")
    return frame.sort_values(["N", "direction"], kind="stable")


DIRECTION_COLORS = {"NNP_to_mW": "#2166ac", "mW_to_NNP": "#d95f02"}
DIRECTION_MARKERS = {"NNP_to_mW": "o", "mW_to_NNP": "s"}
SERIES_NOTE = ("Convergence of the finite sample's Taylor series to that same\n"
               "sample's full exponentially reweighted mean (not a population uncertainty)")


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_signed_coefficients(frame, threshold, output_dir):
    """Signed C1-C4 versus N with bootstrap error bars, one row per direction.

    Filled markers: |C_k|/SE >= threshold (statistically resolved). Open
    markers: not resolved. Each order shares one y-scale across directions.
    """
    sizes = sorted(frame["N"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6), sharex=True, sharey="col")
    for row_index, (direction, label) in enumerate(DIRECTIONS):
        subset = frame.loc[frame["direction"] == direction].sort_values("N")
        color = DIRECTION_COLORS[direction]
        marker = DIRECTION_MARKERS[direction]
        for column, order in enumerate(range(1, 5)):
            ax = axes[row_index, column]
            values = subset[f"C{order}"].to_numpy()
            errors = subset[f"C{order}_se"].to_numpy()
            resolved = subset[f"C{order}_abs_over_se"].to_numpy() >= threshold
            ax.axhline(0, color="0.45", lw=0.8)
            ax.errorbar(subset["N"], values, yerr=errors, fmt="none", ecolor=color,
                        capsize=3, lw=1.2)
            ax.plot(subset["N"], values, color=color, lw=1.0, alpha=0.5)
            ax.plot(subset["N"][resolved], values[resolved], ls="none", marker=marker,
                    ms=7, color=color)
            ax.plot(subset["N"][~resolved], values[~resolved], ls="none", marker=marker,
                    ms=7, color=color, mfc="white", mew=1.4)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(n) for n in sizes], fontsize=8)
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            if row_index == 0:
                ax.set_title(rf"$C_{order}$", fontsize=12)
            if row_index == 1:
                ax.set_xlabel("N")
            if column == 0:
                ax.set_ylabel(f"{label}\n" + r"$C_k$ ± SE [eV/particle]")
    handles = [
        plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=7,
                   label=rf"resolved: $|C_k|/\mathrm{{SE}} \geq {threshold:g}$"),
        plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=7, mfc="white",
                   label=rf"not resolved: $|C_k|/\mathrm{{SE}} < {threshold:g}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Signed cumulant coefficients with bootstrap standard errors "
                 "(frame resampling)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, output_dir / "cumulant_coefficients_signed_bootstrap")


def plot_series_truncation(series, direction, label, output_dir):
    """|G_m - G_full| through the highest saved order, one curve per N."""
    subset = series.loc[series["direction"] == direction]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = plt.get_cmap("tab10")
    # Partial sums can agree with G_full to machine precision; draw those
    # points at a floor instead of dropping them from the log axis.
    floor = 1e-16
    for index, (size, group) in enumerate(subset.groupby("N", sort=True)):
        group = group.sort_values("order")
        values = np.maximum(group["error_vs_full_reweighting"].abs().to_numpy(), floor)
        ax.plot(group["order"], values, marker=MARKERS[index % len(MARKERS)],
                color=colors(index), lw=1.6, ms=5, label=f"N = {int(size)}")
    orders = sorted(subset["order"].unique())
    ax.set_xticks(orders)
    ax.set_xlabel("Expansion order $m$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 2)
    ax.set_ylabel(r"$|G_m - G_{\mathrm{full}}|$ (eV/particle)")
    ax.set_title(f"{label}: empirical Taylor series through order {orders[-1]}")
    ax.text(0.01, 0.01, SERIES_NOTE, transform=ax.transAxes, fontsize=8,
            color="0.3", va="bottom")
    ax.grid(True, which="both", alpha=0.25)
    # Outside the axes: the N=2048 curve stays near the top at every order.
    ax.legend(title="System size", frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, output_dir / f"truncation_error_{direction}_order{orders[-1]}")


def main():
    args = parse_args()
    try:
        frame = load_summary(args.input.resolve())
        series = pd.read_csv(args.series_input.resolve())
        output_dir = args.output_dir.resolve()
        plot_signed_coefficients(frame, args.resolved_threshold, output_dir)
        for direction, label in DIRECTIONS:
            plot_series_truncation(series, direction, label, output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
