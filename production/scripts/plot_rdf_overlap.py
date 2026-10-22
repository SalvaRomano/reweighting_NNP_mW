#!/usr/bin/env python3
"""Direct and reweighted RDFs with residual panels for two system sizes.

Top row: g(r) of direct NNP, direct mW, and both reweighted estimates.
Bottom row: the same signed differences as the coordination-number figure,

    direct NNP - direct mW, NNP -> mW - direct mW, mW -> NNP - direct NNP,

so that agreement which is invisible in g(r) can be judged quantitatively.
The style follows the publication figures in figures/Production/.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

NNP_COLOR = "#2166ac"
MW_COLOR = "#d95f02"
DIRECT_COLOR = "#343a40"
# (column, label, color, linestyle) of the g(r) curves.
CURVES = (
    ("g_NNP", "direct NNP", NNP_COLOR, "-"),
    ("g_mW", "direct mW", MW_COLOR, "-"),
    ("g_NNP_to_mW", "NNP → mW", NNP_COLOR, "--"),
    ("g_mW_to_NNP", "mW → NNP", MW_COLOR, "--"),
)
# (minuend, subtrahend, label, color) of the residual curves.
RESIDUALS = (
    ("g_NNP", "g_mW", "NNP − mW", DIRECT_COLOR),
    ("g_NNP_to_mW", "g_mW", "NNP→mW − mW", NNP_COLOR),
    ("g_mW_to_NNP", "g_NNP", "mW→NNP − NNP", MW_COLOR),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs=2, default=(256, 2048),
                        help="Two system sizes, one per column (default: 256 2048)")
    parser.add_argument("--rdf-dir", type=Path, default=ROOT / "results/rdf_all_sizes")
    parser.add_argument("--rdf-N256", type=Path, default=ROOT / "analysis/rdf_N256.csv")
    parser.add_argument("--rmin", type=float, default=2.0)
    parser.add_argument("--rmax", type=float, default=9.85,
                        help="Upper r limit in Å; the N=64/512 boxes end near 7.8 Å "
                             "(default: 9.85, half the N=256 box)")
    parser.add_argument("--residual-rebin", type=int, default=5,
                        help="Adjacent 0.02 Å bins merged for the residual panels "
                             "(default: 5, i.e. 0.1 Å bins; 1 keeps the original bins)")
    parser.add_argument("--figure-prefix", type=Path,
                        default=ROOT / "figures/Structural/rdf_overlap")
    return parser.parse_args()


def load(n, args):
    path = args.rdf_N256 if n == 256 else args.rdf_dir / f"rdf_N{n}.csv"
    curves = pd.read_csv(path)
    missing = {"r", *(c for c, *_ in CURVES)} - set(curves.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if curves["r"].max() < args.rmax - 0.05:
        raise ValueError(f"N={n}: RDF ends at {curves['r'].max():.2f} Å < rmax")
    return curves[(curves["r"] >= args.rmin) & (curves["r"] <= args.rmax)]


def rebin(r, values, factor):
    """Merge `factor` adjacent bins into one, weighting by shell volume.

    This equals the RDF of a histogram with a factor-times wider bin. For a
    uniform bin width dr, the shell volume is proportional to 3 r^2 dr + dr^3 / 4.
    Incomplete trailing bins are dropped.
    """
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    count = len(r) // factor * factor
    dr = r[1] - r[0]
    volume = (3 * r[:count] ** 2 * dr + dr ** 3 / 4).reshape(-1, factor)
    merged = (values[:count].reshape(-1, factor) * volume).sum(1) / volume.sum(1)
    centers = r[:count].reshape(-1, factor).mean(1)
    return centers, merged


def style_axis(ax):
    """Tick, frame, and grid settings shared with figures/Production/."""
    ax.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(True, which="major", alpha=0.3)


def main():
    args = parse_args()
    # One y scale per row for both sizes, so they compare directly; the
    # second column therefore shows no y tick labels.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.1), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": (2.0, 1.25), "hspace": 0.06,
                                          "wspace": 0.05})
    for column, n in enumerate(args.sizes):
        curves = load(n, args)
        r = curves["r"]
        top, bottom = axes[0, column], axes[1, column]

        # Direct curves are drawn wider underneath the dashed reweighted ones,
        # so overlapping curves stay distinguishable.
        for name, label, color, linestyle in CURVES:
            width = 1.6 if linestyle == "-" else 1.0
            top.plot(r, curves[name], color=color, ls=linestyle, lw=width, label=label)
        top.text(0.97, 0.93, f"N = {n}", transform=top.transAxes,
                 ha="right", va="top", fontsize=9)
        top.set_ylim(bottom=0)
        top.yaxis.set_major_locator(MultipleLocator(0.5))
        top.yaxis.set_minor_locator(MultipleLocator(0.25))

        # Per-bin noise of the 0.02 Å histogram hides systematic differences,
        # so residuals use wider bins (see rebin).
        bottom.axhline(0, color="0.45", lw=0.7, ls="--", zorder=1)
        for minuend, subtrahend, label, color in RESIDUALS:
            r_coarse, delta = rebin(r, curves[minuend] - curves[subtrahend],
                                    args.residual_rebin)
            bottom.plot(r_coarse, delta, color=color, lw=1.0, label=label)

        # Symmetric limits leave headroom for the one-row residual legend.
        bottom.set_ylim(-0.036, 0.036)
        bottom.yaxis.set_major_locator(MultipleLocator(0.02))
        bottom.yaxis.set_minor_locator(MultipleLocator(0.01))
        bottom.set_xlabel(r"$r$ (Å)", fontsize=9)
        bottom.xaxis.set_major_locator(MultipleLocator(2))
        bottom.xaxis.set_minor_locator(MultipleLocator(1))
        bottom.set_xlim(args.rmin, args.rmax)
        for ax in (top, bottom):
            style_axis(ax)

    axes[0, 0].set_ylabel(r"$g(r)$", fontsize=9)
    axes[1, 0].set_ylabel(r"$\Delta g(r)$", fontsize=9)
    # Each legend sits in the row it describes: the g(r) curve styles in the
    # empty region above g(r) = 1, the residual colors as one row along the
    # top of the (quieter) first residual panel. Both apply to both columns.
    axes[0, 0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86),
                      fontsize=6.8, handlelength=2.1)
    # 6.0 pt as in the bias figure of figures/Production/; 6.8 pt does not fit
    # three entries in one row of a half-width panel.
    residual_legend = axes[1, 0].legend(frameon=False, loc="upper center", ncol=3, fontsize=6.0,
                      handlelength=0.9, handletextpad=0.3, columnspacing=0.7,
                      borderaxespad=0.2, borderpad=0.1)
    fig.align_ylabels(axes[:, 0])
    # Guard against the legend spilling outside its panel.
    fig.canvas.draw()
    box = residual_legend.get_window_extent()
    panel = axes[1, 0].get_window_extent()
    if box.x0 < panel.x0 or box.x1 > panel.x1:
        raise ValueError("Residual legend is wider than its panel")

    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.figure_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.figure_prefix}.png and .pdf")


if __name__ == "__main__":
    main()
