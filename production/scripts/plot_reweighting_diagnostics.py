#!/usr/bin/env python3
"""Plot reweighting overlap and validation diagnostics versus system size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


REQUIRED_COLUMNS = {
    "N",
    "neff_fraction_NNP_to_mW",
    "neff_fraction_mW_to_NNP",
    "max_weight_fraction_NNP_to_mW",
    "max_weight_fraction_mW_to_NNP",
    "top10_weight_fraction_NNP_to_mW",
    "top10_weight_fraction_mW_to_NNP",
    "error_NNP_to_mW",
    "error_mW_to_NNP",
    "abs_error_before_NNP_to_mW",
    "abs_error_after_NNP_to_mW",
    "abs_error_before_mW_to_NNP",
    "abs_error_after_mW_to_NNP",
}
COLORS = {"NNP to mW": "tab:blue", "mW to NNP": "tab:orange"}
BEFORE_COLORS = {"NNP to mW": "#8dbce3", "mW to NNP": "#f6b578"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "analysis" / "energy_averages_and_reweighting.csv",
        help="Reweighting summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "figures" / "Reweighting",
        help="Directory for PNG and PDF figures",
    )
    parser.add_argument(
        "--production-absolute-error-only",
        action="store_true",
        help="Create only the publication-styled before/after absolute-error figure",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    """Load, validate, and sort the diagnostic summary."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values")
    if (frame["N"] <= 0).any():
        raise ValueError("N must be positive")
    if frame["N"].duplicated().any():
        duplicates = sorted(frame.loc[frame["N"].duplicated(keep=False), "N"].unique())
        raise ValueError(f"N must be unique in the summary; duplicates: {duplicates}")

    fraction_columns = [column for column in REQUIRED_COLUMNS if "fraction" in column]
    for column in fraction_columns:
        if ((frame[column] < 0.0) | (frame[column] > 1.0)).any():
            raise ValueError(f"{column} contains values outside [0, 1]")
    for column in ["neff_fraction_NNP_to_mW", "neff_fraction_mW_to_NNP"]:
        if (frame[column] <= 0.0).any():
            raise ValueError(f"{column} must be positive for a logarithmic axis")
    return frame.sort_values("N", kind="stable").reset_index(drop=True)


def style_axis(ax: plt.Axes, sizes: np.ndarray, ylabel: str) -> None:
    """Apply common labels and a readable base-two system-size axis."""
    ax.set_xlabel("System size, N")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, bottom_margin: float = 0.0,
    fixed_canvas: bool = False,
) -> None:
    """Save one figure in both high-resolution PNG and vector PDF formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    save_options = {} if fixed_canvas else {"bbox_inches": "tight"}
    fig.savefig(png_path, dpi=300, **save_options)
    fig.savefig(pdf_path, **save_options)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_pair(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    log_y: bool = False,
    unit_interval: bool = False,
) -> None:
    """Plot the two reweighting directions with consistent styling."""
    sizes = frame["N"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        sizes,
        frame[first_column],
        marker="o",
        linewidth=1.8,
        markersize=5,
        color=COLORS["NNP to mW"],
        label="NNP → mW",
    )
    ax.plot(
        sizes,
        frame[second_column],
        marker="s",
        linewidth=1.8,
        markersize=5,
        color=COLORS["mW to NNP"],
        label="mW → NNP",
    )
    if log_y:
        ax.set_yscale("log")
    if unit_interval:
        # Fractions live in [0, 1]; slight headroom keeps markers at one from
        # being clipped by the upper axis spine.
        ax.set_ylim(0.0, 1.02)
    style_axis(ax, sizes, ylabel)
    save_figure(fig, output_dir, stem)


def plot_before_after_errors(frame: pd.DataFrame, output_dir: Path) -> None:
    """Compare both directions and stages on a shared per-particle error axis."""
    sizes = frame["N"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for direction, label in (("NNP_to_mW", "NNP → mW"),
                             ("mW_to_NNP", "mW → NNP")):
        color_key = "NNP to mW" if direction == "NNP_to_mW" else "mW to NNP"
        for stage, marker, linestyle, color in (
            ("before", "o", "--", BEFORE_COLORS[color_key]),
            ("after", "s", "-", COLORS[color_key]),
        ):
            ax.plot(
                sizes,
                frame[f"abs_error_{stage}_{direction}"],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                color=color,
                label=f"{label}: {stage}",
            )
    style_axis(ax, sizes, "Absolute error (eV/particle)")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    # U is total potential energy; only the observable u is per particle.
    fig.text(
        0.11, 0.085,
        r"$u_t=U_t/N,\quad E_{\rm before}=|\langle u_t\rangle_s-\langle u_t\rangle_t|$",
        fontsize=10,
    )
    fig.text(
        0.11, 0.045,
        r"$E_{\rm after}=|\langle u_t\rangle_{s,{\rm rw}}-\langle u_t\rangle_t|,"
        r"\quad w_i\propto e^{-\beta(U_{t,i}-U_{s,i})}$",
        fontsize=10,
    )
    fig.text(0.11, 0.015,
             "s: source ensemble; t: target Hamiltonian; U: total potential energy",
             fontsize=8)
    save_figure(fig, output_dir, "absolute_error_before_after_both_directions_vs_N",
                bottom_margin=0.17)


def plot_production_before_after_errors(frame: pd.DataFrame, output_dir: Path) -> None:
    """Create the uncluttered, publication-styled before/after error figure."""
    sizes = frame["N"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for direction, label in (("NNP_to_mW", "NNP → mW"),
                             ("mW_to_NNP", "mW → NNP")):
        color_key = "NNP to mW" if direction == "NNP_to_mW" else "mW to NNP"
        for stage, marker, linestyle, color in (
            ("before", "o", "--", BEFORE_COLORS[color_key]),
            ("after", "s", "-", COLORS[color_key]),
        ):
            ax.plot(
                sizes,
                frame[f"abs_error_{stage}_{direction}"],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.35,
                markersize=4.2,
                color=color,
                label=f"{label}: {stage}",
            )

    ax.set_xlabel("System size, N", fontsize=9)
    ax.set_ylabel("Absolute energy error (eV/particle)", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.tick_params(axis="both", which="major", labelsize=8, length=4, width=0.9)
    ax.yaxis.get_offset_text().set_fontsize(8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.grid(True, which="major", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib fills legend columns vertically. This order places both blue
    # entries on the first row and both orange entries on the second row.
    order = [0, 2, 1, 3]
    ax.legend([handles[index] for index in order], [labels[index] for index in order],
              frameon=False, ncol=2, loc="upper left", fontsize=6.8,
              handlelength=2.1, columnspacing=0.8)
    save_figure(fig, output_dir, "absolute_error_before_after_both_directions_vs_N",
                fixed_canvas=True)


def make_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Create all requested overlap and validation plots."""
    plot_pair(
        frame,
        "neff_fraction_NNP_to_mW",
        "neff_fraction_mW_to_NNP",
        "Effective sample size fraction",
        output_dir,
        "neff_fraction_vs_N",
        log_y=True,
    )
    plot_pair(
        frame,
        "max_weight_fraction_NNP_to_mW",
        "max_weight_fraction_mW_to_NNP",
        "Largest normalized weight fraction",
        output_dir,
        "max_weight_fraction_vs_N",
        log_y=True,
    )
    plot_pair(
        frame,
        "top10_weight_fraction_NNP_to_mW",
        "top10_weight_fraction_mW_to_NNP",
        "Top-10 normalized weight fraction",
        output_dir,
        "top10_weight_fraction_vs_N",
        unit_interval=True,
    )

    error_frame = frame.copy()
    error_frame["abs_error_NNP_to_mW"] = error_frame["error_NNP_to_mW"].abs()
    error_frame["abs_error_mW_to_NNP"] = error_frame["error_mW_to_NNP"].abs()
    plot_pair(
        error_frame,
        "abs_error_NNP_to_mW",
        "abs_error_mW_to_NNP",
        "Absolute reweighting error (eV/particle)",
        output_dir,
        "reweighting_error_vs_N",
    )
    plot_before_after_errors(frame, output_dir)


def main() -> int:
    """Run the plotting workflow."""
    args = parse_args()
    try:
        frame = load_summary(args.input.resolve())
        if args.production_absolute_error_only:
            plot_production_before_after_errors(frame, args.output_dir.resolve())
        else:
            make_plots(frame, args.output_dir.resolve())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
