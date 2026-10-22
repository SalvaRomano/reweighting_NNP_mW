#!/usr/bin/env python3
"""Signed first-shell coordination differences with block-bootstrap errors.

For every system size, the coordination number at the common cutoff is
computed frame by frame from the LAMMPS dumps. Frames and reweighting weights
are matched exactly as in compute_rdf_N256.py. The script reports

    direct    : CN(NNP) - CN(mW)
    NNP -> mW : CN(NNP reweighted to mW) - CN(mW)
    mW -> NNP : CN(mW reweighted to NNP) - CN(NNP)

Signed differences within one size cancel the box-density dependence of the
absolute coordination number. A successful reweighting brings the last two
differences to zero within error.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.compute_rdf_N256 as rdf_pipeline  # noqa: E402
from src.coordination import (bootstrap_differences,  # noqa: E402
                              coordination_differences, frame_coordination)
from src.rdf_features import COMMON_CUTOFF_A  # noqa: E402


SIZES = (64, 128, 256, 512, 1024, 2048)
SERIES = (
    ("delta_direct", "direct NNP − direct mW", "#343a40", "o"),
    ("delta_NNP_to_mW", "NNP → mW − direct mW", "#2166ac", "^"),
    ("delta_mW_to_NNP", "mW → NNP − direct NNP", "#d95f02", "v"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--trajectory-root", type=Path, default=ROOT / "lammps")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/rdf_all_sizes")
    parser.add_argument("--figure-prefix", type=Path,
                        default=ROOT / "figures/Structural/coordination_differences_all_sizes")
    parser.add_argument("--cutoff", type=float, default=COMMON_CUTOFF_A)
    parser.add_argument("--temperature", type=float, default=300.0)
    # Saved frames are nearly uncorrelated: the block-averaged standard error
    # stays within ~1.0-1.5x the naive one up to 100-frame blocks. 50 frames
    # is a conservative block length.
    parser.add_argument("--block-length", type=int, default=50,
                        help="Frames per contiguous bootstrap block (default: 50)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap replicates (default: 2000)")
    parser.add_argument("--seed", type=int, default=20261013,
                        help="Bootstrap random seed (default: 20261013)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute per-frame values even if the cache exists")
    return parser.parse_args()


def per_frame_size(n, args):
    """Per-frame coordination and normalized weights for one system size."""
    # The RDF pipeline keeps the size as a module global; reuse its frame
    # matching, endpoint exclusion, and weights unchanged.
    rdf_pipeline.N = n
    _, _, weight_maps, _ = rdf_pipeline.load_master(args.input.resolve(), args.temperature)
    rows = []
    for model, run, path in rdf_pipeline.trajectory_paths(args.trajectory_root.resolve()):
        for step, count, lower, lengths, xyz in rdf_pipeline.read_dump(path, positions=True):
            weight = weight_maps[model].get((run, step))
            if weight is None:
                continue  # endpoint or frame absent from the master table
            if count != n:
                raise ValueError(f"N={n}: unexpected atom count at {model} run {run} step {step}")
            rows.append((n, model, run, step,
                         frame_coordination(xyz, lower, lengths, args.cutoff), weight))
    frame = pd.DataFrame(rows, columns=["N", "model", "source_run", "step", "cn", "weight"])
    frame["r_cut"] = args.cutoff
    for model in ("NNP", "mW"):
        if (frame["model"] == model).sum() != len(weight_maps[model]):
            raise ValueError(f"N={n}: missing {model} frames in the trajectories")
    print(f"N={n}: {(frame.model == 'NNP').sum()} NNP and "
          f"{(frame.model == 'mW').sum()} mW frames", flush=True)
    return frame


def ensemble(frame, model):
    """Time-ordered arrays of one model's frames for the bootstrap."""
    part = frame[frame["model"] == model].sort_values(["source_run", "step"])
    return {"cn": part["cn"].to_numpy(), "weight": part["weight"].to_numpy(),
            "run": part["source_run"].to_numpy()}


def summarize(frames, args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in SIZES:
        frame = frames[frames["N"] == n]
        nnp, mw = ensemble(frame, "NNP"), ensemble(frame, "mW")
        estimate = coordination_differences(nnp, mw)
        replicates = bootstrap_differences(nnp, mw, args.block_length, args.bootstrap, rng)
        row = {"N": n, "n_frames_NNP": len(nnp["cn"]), "n_frames_mW": len(mw["cn"]),
               "cn_NNP": nnp["cn"].mean(), "cn_mW": mw["cn"].mean(),
               "neff_NNP_to_mW": 1 / np.sum(nnp["weight"] ** 2),
               "neff_mW_to_NNP": 1 / np.sum(mw["weight"] ** 2)}
        for key in estimate:
            se = replicates[key].std(ddof=1)
            row[key] = estimate[key]
            row[f"{key}_se"] = se
            row[f"{key}_z"] = estimate[key] / se
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.insert(1, "r_cut", args.cutoff)
    summary["block_length"] = args.block_length
    summary["bootstrap"] = args.bootstrap
    summary["seed"] = args.seed
    return summary


def plot(summary, prefix):
    # Single-column style shared with the figures in figures/Production/.
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    x = np.arange(len(summary))
    for offset, (key, label, color, marker) in zip((-0.15, 0, 0.15), SERIES):
        ax.errorbar(x + offset, summary[key], yerr=summary[f"{key}_se"],
                    color=color, marker=marker, ms=4.2, lw=0, elinewidth=0.9,
                    capsize=2, capthick=0.9, label=label, zorder=3)
    ax.axhline(0, color="0.45", lw=0.7, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in summary["N"]])
    ax.set_xlabel("System size, N", fontsize=9)
    ax.set_ylabel(r"$\Delta N_c$", fontsize=9)
    # Labelled major ticks every 5e-3 with one unlabelled minor tick between,
    # and a shared x10^-3 offset, so the axis does not list every value.
    ax.yaxis.set_major_locator(MultipleLocator(5e-3))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5e-3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.9)
    ax.tick_params(axis="y", which="minor", length=2, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(frameon=False, loc="lower left", fontsize=6.8, handletextpad=0.3)
    fig.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache = args.results_dir / "coordination_per_frame.csv"
    if cache.is_file() and not args.recompute:
        frames = pd.read_csv(cache)
        if not np.allclose(frames["r_cut"], args.cutoff):
            raise ValueError("Cached cutoff differs; rerun with --recompute")
        print(f"Loaded per-frame coordination from {cache}")
    else:
        frames = pd.concat([per_frame_size(n, args) for n in SIZES], ignore_index=True)
        frames.to_csv(cache, index=False, float_format="%.12g")
        print(f"Saved {cache}")

    summary = summarize(frames, args)
    output = args.results_dir / "coordination_differences.csv"
    summary.to_csv(output, index=False, float_format="%.8g")
    plot(summary, args.figure_prefix)

    columns = ["N", "cn_NNP", "cn_mW", "neff_NNP_to_mW", "neff_mW_to_NNP"]
    for key, *_ in SERIES:
        columns += [key, f"{key}_se", f"{key}_z"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Saved {output}")
    print(f"Saved {args.figure_prefix}.png and .pdf")


if __name__ == "__main__":
    main()
