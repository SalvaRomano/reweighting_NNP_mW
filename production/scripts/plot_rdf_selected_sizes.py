#!/usr/bin/env python3
"""Plot saved direct and reweighted RDF curves for selected system sizes."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(1024, 2048))
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "results/rdf_all_sizes")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/Structural")
    return parser.parse_args()


def plot_rdf(path, size, prefix):
    curves = pd.read_csv(path)
    required = {"r", "g_NNP", "g_mW", "g_NNP_to_mW", "g_mW_to_NNP"}
    if not required.issubset(curves.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(curves.columns))}")

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    styles = (
        ("g_NNP", "NNP direct", "tab:blue", "-"),
        ("g_mW", "mW direct", "tab:orange", "-"),
        ("g_NNP_to_mW", "NNP → mW reweighted", "tab:blue", "--"),
        ("g_mW_to_NNP", "mW → NNP reweighted", "tab:orange", "--"),
    )
    for column, label, color, linestyle in styles:
        ax.plot(curves["r"], curves[column], color=color, linestyle=linestyle,
                linewidth=1.8, label=label)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, curves["r"].max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"Particle-particle RDF, N = {size}")
    fig.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output = prefix.with_suffix(suffix)
        fig.savefig(output, dpi=300 if suffix == ".png" else None,
                    bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)


def main():
    args = parse_args()
    for size in args.sizes:
        plot_rdf(args.input_dir / f"rdf_N{size}.csv", size,
                 args.figure_dir / f"rdf_N{size}")


if __name__ == "__main__":
    main()
