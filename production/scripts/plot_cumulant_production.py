#!/usr/bin/env python3
"""Publication versions of the cumulant-coefficient figures.

Writes, without touching the diagnostic figures of plot_cumulant_expansion.py:

- cumulant_coefficients_bootstrap: signed C1-C4 versus N with bootstrap
  standard errors; filled markers have |C_k|/SE >= threshold.
- cumulant_significance: |C_k|/SE for k = 1-4, grouped by N in increasing
  order, with the two directions stacked vertically.

It also prints the order-12 residual of the empirical Taylor series for every
N and direction as a sanity check.
"""

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
from matplotlib.ticker import ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDERS = (1, 2, 3, 4)
# Same direction colors and labels as the manuscript figures in
# plot_reweighting_diagnostics.py.
DIRECTIONS = (
    ("NNP_to_mW", "NNP → mW", "tab:blue", "o"),
    ("mW_to_NNP", "mW → NNP", "tab:orange", "s"),
)
ORDER_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 7.5
LINE_WIDTH = 1.2
MARKER_SIZE = 4.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulant_expansion/summary.csv")
    parser.add_argument("--series-input", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/Cumulant/production")
    parser.add_argument("--resolved-threshold", type=float, default=3.0)
    return parser.parse_args()


def load_summary(path):
    frame = pd.read_csv(path)
    required = {"N", "direction"}
    for k in ORDERS:
        required.update({f"C{k}", f"C{k}_se", f"C{k}_abs_over_se"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if frame.duplicated(["N", "direction"]).any():
        raise ValueError("Duplicate N/direction rows")
    if set(frame["direction"]) != {d for d, *_ in DIRECTIONS}:
        raise ValueError("Both reweighting directions are required")
    values = frame[sorted(required - {"direction"})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values in the summary")
    # The stored ratio must agree with the plotted C and SE columns.
    for k in ORDERS:
        ratio = frame[f"C{k}"].abs() / frame[f"C{k}_se"]
        if not np.allclose(ratio, frame[f"C{k}_abs_over_se"], rtol=1e-6):
            raise ValueError(f"C{k}_abs_over_se is inconsistent with C{k}/C{k}_se")
    return frame


def style_axis(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, length=3.5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(True, which="major", alpha=0.25, lw=0.6)


def size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 1.35)


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=600 if suffix == ".png" else None)
        print(f"Saved {path}")
    plt.close(fig)


def plot_coefficients(frame, threshold, stem):
    sizes = sorted(int(n) for n in frame["N"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7), sharex=True, sharey="col",
                             layout="constrained")
    for column, k in enumerate(ORDERS):
        # One power of ten per order, chosen from that order's largest |C|+SE,
        # and stated once in the column header.
        extent = (frame[f"C{k}"].abs() + frame[f"C{k}_se"]).max()
        exponent = int(np.floor(np.log10(extent)))
        scale = 10.0 ** exponent
        for row, (direction, label, color, marker) in enumerate(DIRECTIONS):
            ax = axes[row, column]
            data = frame.loc[frame["direction"] == direction].sort_values("N")
            n = data["N"].to_numpy(dtype=float)
            value = data[f"C{k}"].to_numpy() / scale
            error = data[f"C{k}_se"].to_numpy() / scale
            resolved = data[f"C{k}_abs_over_se"].to_numpy() >= threshold
            ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)
            ax.plot(n, value, color=color, lw=0.9, alpha=0.45, zorder=2)
            ax.errorbar(n, value, yerr=error, fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=2.2, capthick=1.0, zorder=3)
            ax.plot(n[resolved], value[resolved], ls="none", marker=marker,
                    ms=MARKER_SIZE, color=color, mec=color, zorder=4)
            ax.plot(n[~resolved], value[~resolved], ls="none", marker=marker,
                    ms=MARKER_SIZE, color=color, mfc="white", mec=color, mew=1.0, zorder=4)
            size_axis(ax, sizes)
            style_axis(ax)
            ax.tick_params(axis="x", labelsize=TICK_SIZE - 1, rotation=45)
            if row == 0:
                ax.set_title(rf"$C_{k}$  [$10^{{{exponent}}}$ eV/particle]",
                             fontsize=LABEL_SIZE, pad=4)
            if column == 0:
                ax.set_ylabel(label, fontsize=LABEL_SIZE)
    fig.supxlabel("System size, N", fontsize=LABEL_SIZE)
    handles = [
        Line2D([], [], color="0.25", marker="o", ls="none", ms=MARKER_SIZE,
               label=rf"$|C_k|/\mathrm{{SE}} \geq {threshold:g}$"),
        Line2D([], [], color="0.25", marker="o", ls="none", ms=MARKER_SIZE, mfc="white",
               label=rf"$|C_k|/\mathrm{{SE}} < {threshold:g}$"),
        Line2D([], [], color="0.25", marker="|", ls="none", ms=7, mew=1.0,
               label="±1 bootstrap SE"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
               fontsize=LEGEND_SIZE, handletextpad=0.3, columnspacing=1.4)
    save(fig, stem)


def plot_significance(frame, threshold, stem):
    sizes = sorted(int(n) for n in frame["N"].unique())
    positions = np.arange(len(sizes))
    offsets = np.linspace(-0.27, 0.27, len(ORDERS))
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.5), sharex=True,
                             layout="constrained")
    for ax, (direction, label, color, _) in zip(axes, DIRECTIONS):
        data = frame.loc[frame["direction"] == direction].set_index("N").loc[sizes]
        ratios = np.column_stack([data[f"C{k}_abs_over_se"].to_numpy() for k in ORDERS])
        for i, position in enumerate(positions):
            # Thin connector groups the four orders of one N.
            ax.plot(position + offsets, ratios[i], color=color, lw=0.8, alpha=0.5, zorder=2)
        for j, k in enumerate(ORDERS):
            x = positions + offsets[j]
            ax.plot(x, ratios[:, j], ls="none", marker="o",
                    ms=MARKER_SIZE, color=color, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        # Scale each direction separately so the smaller ratios remain readable.
        ax.set_ylim(0, ratios.max() * 1.08)
        ax.text(0.98, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=LABEL_SIZE)
        style_axis(ax)
        ax.grid(False, axis="x")
    axes[-1].set_xlabel("System size, N", fontsize=LABEL_SIZE)
    fig.supylabel(r"$|C_k|\,/\,\mathrm{SE}(C_k)$", fontsize=LABEL_SIZE)
    save(fig, stem)


def print_order12_residuals(path):
    series = pd.read_csv(path)
    last = series["order"].max()
    rows = series.loc[series["order"] == last, ["N", "direction", "order",
                                                "error_vs_full_reweighting"]]
    rows = rows.sort_values(["direction", "N"])
    print(f"\nResidual G_{last} - G_full of the empirical series [eV/particle]:")
    print(rows.to_string(index=False, float_format=lambda v: f"{v: .3e}"))


def main():
    args = parse_args()
    try:
        frame = load_summary(args.input.resolve())
        output = args.output_dir.resolve()
        plot_coefficients(frame, args.resolved_threshold,
                          output / "cumulant_coefficients_bootstrap")
        plot_significance(frame, args.resolved_threshold, output / "cumulant_significance")
        print_order12_residuals(args.series_input.resolve())
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
