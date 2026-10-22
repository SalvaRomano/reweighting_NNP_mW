#!/usr/bin/env python3
"""Two-direction order-six truncation figure at manuscript column width."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, VPacker, TextArea
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = (("NNP_to_mW", "NNP → mW", "o"), ("mW_to_NNP", "mW → NNP", "s"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/cumulant_truncation_bootstrap.csv")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/Cumulant/production")
    parser.add_argument("--width", type=float, default=3.45)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, float_precision="round_trip")
    frame = frame.loc[frame.order <= 6].set_index(["N", "direction", "order"])
    reference = pd.read_csv(args.reference, float_precision="round_trip")
    reference = reference.set_index(["N", "direction", "order"])
    sizes = sorted(frame.index.get_level_values("N").unique())
    expected = {(n, d, m) for n in sizes for d, _, _ in DIRECTIONS for m in range(7)}
    if frame.index.has_duplicates or set(frame.index) != expected:
        raise ValueError("Missing or duplicate size/direction/order rows")
    np.testing.assert_array_equal(frame.original_epsilon.to_numpy(),
                                  reference.loc[frame.index, "error_vs_full_reweighting"].abs())
    columns = ["original_epsilon", "bootstrap_median_epsilon",
               "bootstrap_epsilon_p16", "bootstrap_epsilon_p84"]
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Residuals and intervals must be finite and nonnegative")
    if ((values[:, 2] > values[:, 1]) | (values[:, 1] > values[:, 3])).any():
        raise ValueError("Percentiles must bracket the bootstrap median")
    print(f"Validated {len(frame)} direction-specific original residuals and intervals")
    colors = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(sizes)}
    style = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 7,
             "ytick.labelsize": 8, "axes.linewidth": 0.9,
             "mathtext.fontset": "cm", "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(args.width, 3.0), sharey=True,
                                 layout="constrained")
        floor = 1e-16
        # Common limits permit direct comparison across the two directions.
        ymin = np.maximum(values[:, [0, 2]], floor).min() / 2
        ymax = np.maximum(values[:, [0, 3]], floor).max() * 2
        for ax, (direction, label, marker) in zip(axes, DIRECTIONS):
            for n in sizes:
                data = frame.loc[[(n, direction, m) for m in range(7)]]
                median = np.maximum(data.bootstrap_median_epsilon.to_numpy(), floor)
                lower = np.maximum(data.bootstrap_epsilon_p16.to_numpy(), floor)
                upper = np.maximum(data.bootstrap_epsilon_p84.to_numpy(), floor)
                # Bars show percentile endpoints around the bootstrap median;
                # the marked line preserves the original empirical residual.
                ax.errorbar(np.arange(7), median,
                            yerr=np.array([median - lower, upper - median]),
                            fmt="none", ecolor=colors[n], elinewidth=0.7,
                            capsize=1.4, capthick=0.7, alpha=0.55, zorder=1)
                ax.plot(np.arange(7), np.maximum(data.original_epsilon, floor),
                        marker=marker, ms=2.7, color=colors[n], lw=0.9, zorder=2)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-0.25, 6.25)
            ax.set_xticks(np.arange(7))
            ax.set_xlabel(r"Expansion order $m$", fontsize=12)
            ax.tick_params(which="major", length=3, width=0.9)
            ax.tick_params(which="minor", length=1.5, width=0.6)
            ax.grid(which="major", alpha=0.25)
            ax.text(0.04, 0.04, label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9)
        # An axes label centers on the plot area, excluding the shared legend.
        axes[0].set_ylabel(r"$\epsilon_m^{\mathrm{tr}}$ (eV/particle)", fontsize=13)
        # State N once so all six sizes fit on one row at the retained font size.
        handles = [Line2D([], [], color=colors[n], lw=1.4,
                          label=rf"$N={n}$" if i == 0 else rf"${n}$")
                   for i, n in enumerate(sizes)]
        legend = fig.legend(handles=handles, loc="outside lower center", ncol=6,
                            frameon=False, fontsize=9, handlelength=0.6,
                            handletextpad=0.2, columnspacing=1.0)
        # Reserve label space with normal text, then overlay separately sized
        # text/math pieces at the resolved label positions. MathText itself
        # does not support a font-size change within a single label string.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        label = axes[0].xaxis.label
        position = fig.transFigure.inverted().transform(
            label.get_transform().transform(label.get_position()))
        position[0] = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        pieces = HPacker(children=[
            TextArea("Expansion order", textprops={"fontsize": 9}),
            TextArea(r"$m$", textprops={"fontsize": 12}),
        ], align="center", pad=0, sep=3)
        x_label = AnnotationBbox(pieces, position, xycoords=fig.transFigure,
                                frameon=False, pad=0, box_alignment=(0.5, 0.5))
        fig.add_artist(x_label)
        for ax in axes:
            ax.xaxis.label.set_alpha(0)
        label = axes[0].yaxis.label
        pieces = VPacker(children=[
            TextArea("(eV/particle)", textprops={"fontsize": 9, "rotation": 90}),
            TextArea(r"$\epsilon_m^{\mathrm{tr}}$",
                     textprops={"fontsize": 13, "rotation": 90}),
        ], align="center", pad=0, sep=3)
        position = fig.transFigure.inverted().transform(
            label.get_transform().transform(label.get_position()))
        axes[0].add_artist(AnnotationBbox(pieces, position,
                                         xycoords=fig.transFigure,
                                         frameon=False, pad=0, box_alignment=(0.5, 0.5)))
        label.set_alpha(0)
        # Place the legend three points below the shared x label.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_bottom = x_label.get_window_extent(renderer).y0
        legend_top = legend.get_window_extent(renderer).y1
        shift = (label_bottom - 3 * fig.dpi / 72 - legend_top) / fig.bbox.height
        shift += 0.025
        legend.set_bbox_to_anchor((0.5, shift), transform=fig.transFigure)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            path = args.output_dir / f"truncation_error_bootstrap_order6_horizontal.{suffix}"
            fig.savefig(path, dpi=600)
            print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
