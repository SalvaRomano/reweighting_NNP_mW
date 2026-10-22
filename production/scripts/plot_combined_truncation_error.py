#!/usr/bin/env python3
"""Combine empirical Taylor truncation errors for both reweighting directions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = (("NNP_to_mW", "NNP → mW", "o"),
              ("mW_to_NNP", "mW → NNP", "s"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/Cumulant")
    parser.add_argument("--bootstrap-input", type=Path,
                        help="Add 16–84%% bootstrap percentile bars from the summary CSV")
    parser.add_argument("--direction", choices=[item[0] for item in DIRECTIONS],
                        help="Plot only this direction in a separate figure")
    parser.add_argument("--max-order", type=int,
                        help="Highest expansion order to display")
    parser.add_argument("--legend-size", type=float, default=6.8,
                        help="Legend font size in points")
    args = parser.parse_args()
    series = pd.read_csv(args.input, float_precision="round_trip")
    required = {"N", "direction", "order", "error_vs_full_reweighting"}
    if required - set(series.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(series.columns))}")
    if series.duplicated(["N", "direction", "order"]).any():
        raise ValueError("Duplicate size/direction/order rows")
    if not np.isfinite(series[list(required - {"direction"})].to_numpy(float)).all():
        raise ValueError("Non-finite series values")
    sizes = sorted(series["N"].unique())
    orders = sorted(series["order"].unique())
    expected = {(size, direction, order) for size in sizes
                for direction, _, _ in DIRECTIONS for order in orders}
    if set(series[["N", "direction", "order"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Both directions must contain every size and order")
    bootstrap = None
    if args.bootstrap_input is not None:
        bootstrap = pd.read_csv(args.bootstrap_input, float_precision="round_trip")
        bootstrap = bootstrap.set_index(["N", "direction", "order"])
        if bootstrap.index.has_duplicates or set(bootstrap.index) != expected:
            raise ValueError("Bootstrap rows must match every plotted size/direction/order")
        columns = ["bootstrap_median_epsilon", "bootstrap_epsilon_p16",
                   "bootstrap_epsilon_p84"]
        values = bootstrap[columns].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Bootstrap intervals must be finite and nonnegative")
        if ((values[:, 1] > values[:, 0]) | (values[:, 0] > values[:, 2])).any():
            raise ValueError("Bootstrap percentiles must bracket their median")
        # Align by explicit keys and check the original residual to prevent
        # accidentally associating errors with another direction or dataset.
        keyed = series.set_index(["N", "direction", "order"])
        np.testing.assert_array_equal(
            bootstrap.loc[keyed.index, "original_epsilon"].to_numpy(),
            keyed["error_vs_full_reweighting"].abs().to_numpy())
        print(f"Validated all {len(keyed)} bootstrap rows against original residuals")
    directions = [item for item in DIRECTIONS
                  if args.direction is None or item[0] == args.direction]
    if args.max_order is not None:
        if args.max_order not in orders:
            raise ValueError("Requested maximum order must exist in the series")
        orders = [order for order in orders if order <= args.max_order]
        series = series.loc[series["order"] <= args.max_order]

    # Match the physical canvas and type sizes of the Bias/Energy production figures.
    style = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
             "ytick.labelsize": 8, "legend.fontsize": args.legend_size,
             "pdf.fonttype": 42, "mathtext.fontset": "cm", "axes.linewidth": 0.9}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.45, 3.0), layout="constrained")
        colors = {size: plt.get_cmap("tab10")(i) for i, size in enumerate(sizes)}
        # Preserve the original plots' floor for residuals at machine precision.
        floor = 1e-16
        displayed_min = np.inf
        displayed_max = 0.0
        for direction, _, marker in directions:
            for size in sizes:
                group = series.loc[(series["direction"] == direction)
                                   & (series["N"] == size)].sort_values("order")
                values = np.maximum(group["error_vs_full_reweighting"].abs(), floor)
                displayed_min = min(displayed_min, float(values.min()))
                displayed_max = max(displayed_max, float(values.max()))
                if bootstrap is not None:
                    data = bootstrap.loc[[(size, direction, order) for order in orders]]
                    median = np.maximum(data["bootstrap_median_epsilon"].to_numpy(), floor)
                    lower = np.maximum(data["bootstrap_epsilon_p16"].to_numpy(), floor)
                    upper = np.maximum(data["bootstrap_epsilon_p84"].to_numpy(), floor)
                    displayed_min = min(displayed_min, float(lower.min()))
                    displayed_max = max(displayed_max, float(upper.max()))
                    # Percentile bars bracket the bootstrap median, not necessarily
                    # the original residual. Offset only their x positions for clarity.
                    offset = 0 if args.direction else (-0.07 if marker == "o" else 0.07)
                    ax.errorbar(np.array(orders) + offset, median,
                                yerr=np.array([median - lower, upper - median]),
                                fmt="none", ecolor=colors[size], elinewidth=0.65,
                                capsize=1.3, capthick=0.65, alpha=0.55, zorder=1)
                ax.plot(group["order"], values, color=colors[size], marker=marker,
                        lw=1.0, ms=3.0, markeredgewidth=0.5, zorder=2)
        # Short order ranges show every label; longer ranges use alternating ticks.
        if args.max_order is not None and args.max_order <= 6:
            ax.set_xticks(orders)
            ax.set_xticks([], minor=True)
        else:
            ax.set_xticks([order for order in orders if order % 2 == 0])
            ax.set_xticks([order for order in orders if order % 2 != 0], minor=True)
        if args.max_order is not None:
            ax.set_xlim(-0.2, args.max_order + 0.2)
        ax.set_xlabel(r"Expansion order $m$")
        ax.set_ylabel(r"$\epsilon_m^{\mathrm{tr}}$ (eV/particle)")
        ax.set_yscale("log")
        if args.max_order is not None:
            # Fit the selected orders, including the full percentile-bar extent.
            # Modest logarithmic padding leaves room around the extreme values.
            ax.set_ylim(displayed_min / 2, displayed_max * 2)
        else:
            ax.set_ylim(bottom=floor / 2)
        ax.tick_params(axis="both", which="major", length=4, width=0.9)
        ax.tick_params(axis="both", which="minor", length=2, width=0.7)
        ax.grid(True, which="major", alpha=0.25)
        # Independent legends keep color tied to size and shape tied to direction.
        size_handles = [Line2D([], [], color=colors[size], lw=1.35,
                               label=rf"$N = {int(size)}$") for size in sizes]
        direction_handles = [Line2D([], [], color="0.25", marker=marker,
                                    ls="none", ms=4.2, label=label)
                             for _, label, marker in directions]
        if bootstrap is not None:
            direction_handles.append(Line2D([], [], color="0.4", marker="|",
                                            ms=6, ls="none", label="16–84%"))
        # Matplotlib fills legend columns vertically: sizes occupy the left
        # column, with directions in the right column aligned at the bottom.
        blank_handles = [Line2D([], [], ls="none", label=" ")
                         for _ in range(len(size_handles) - len(direction_handles))]
        ax.legend(handles=size_handles + blank_handles + direction_handles,
                  ncol=2, loc="lower left", frameon=False, columnspacing=0.8,
                  handletextpad=0.5, borderaxespad=0.5, handlelength=1.3)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = f"truncation_error_{args.direction or 'combined'}"
        if bootstrap is not None:
            name += "_bootstrap"
        if args.max_order is not None:
            name += f"_order{args.max_order}"
        stem = args.output_dir / name
        for suffix in (".pdf", ".png"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=600, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
