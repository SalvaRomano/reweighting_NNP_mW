#!/usr/bin/env python3
"""Plot thermodynamic observables against concatenated configuration index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_COLUMNS = {"model", "N", "run", "step"}
QUANTITIES = [
    ("temperature", "Temperature (K)"),
    ("poteng_per_particle", "Source potential energy (eV/particle)"),
    ("cross_poteng_per_particle", "Cross potential energy (eV/particle)"),
    ("delta_u", r"$\Delta u$ (eV/particle)"),
    ("toteng_per_particle", "Total energy (eV/particle)"),
    ("press", "Pressure"),
    ("volume", "Volume"),
    ("density", "Density"),
]
COLORS = {"NNP": "tab:blue", "mW": "tab:orange"}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "analysis" / "thermo_all_with_cross_merged.csv",
        help="Input thermo CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "figures" / "Reweighting" / "thermo_vs_configuration",
        help="Directory for per-N PNG and PDF figures",
    )
    return parser.parse_args()


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load data, trim each run's endpoints, and derive intensive energies."""
    frame = pd.read_csv(path)
    missing = sorted(BASE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame["model"] = frame["model"].astype(str)
    numeric_columns = ["N", "run", "step"]
    if "source_run" in frame:
        numeric_columns.append("source_run")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("N, run, source_run (when present), and step must be finite")
    if (frame["N"] <= 0).any():
        raise ValueError("N must be positive")

    # The merged NNP rows all have run=0; source_run still identifies the five
    # original trajectories whose endpoints must be removed separately.
    run_column = "source_run" if "source_run" in frame else "run"
    group_columns = ["model", "N", run_column]
    counts = frame.groupby(group_columns, sort=True).size()
    if (counts < 3).any():
        raise ValueError("Every model/N/run group needs at least three configurations")
    grouped_steps = frame.groupby(group_columns, sort=False)["step"]
    first_step = grouped_steps.transform("min")
    last_step = grouped_steps.transform("max")
    frame = frame.loc[(frame["step"] != first_step) & (frame["step"] != last_step)].copy()

    # Derive per-particle energies from primary columns for comparable scales.
    if "poteng" in frame:
        frame["poteng_per_particle"] = frame["poteng"] / frame["N"]
    if "cross_poteng" in frame:
        frame["cross_poteng_per_particle"] = frame["cross_poteng"] / frame["N"]
        if "poteng" in frame:
            frame["delta_u"] = (frame["cross_poteng"] - frame["poteng"]) / frame["N"]
    if "toteng" in frame:
        frame["toteng_per_particle"] = frame["toteng"] / frame["N"]

    candidates = [(column, label) for column, label in QUANTITIES if column in frame]
    for column, _ in candidates:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    available = [
        (column, label) for column, label in candidates if frame[column].notna().any()
    ]
    if not available:
        raise ValueError("No supported thermodynamic quantity columns contain data")
    return frame, available


def concatenate_runs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Sort runs side by side and return their internal boundary positions."""
    run_column = "source_run" if "source_run" in frame else "run"
    ordered = frame.sort_values([run_column, "step"], kind="stable").copy()
    ordered["configuration_index"] = np.arange(len(ordered), dtype=np.int64)
    run_sizes = ordered.groupby(run_column, sort=True).size().to_numpy(dtype=int)
    boundaries = np.cumsum(run_sizes)[:-1].tolist()
    return ordered, boundaries


def plot_system_size(
    frame: pd.DataFrame,
    size: int,
    quantities: list[tuple[str, str]],
    output_dir: Path,
) -> None:
    """Create one multi-panel thermodynamic trace figure for a system size."""
    n_columns = 2
    n_rows = int(np.ceil(len(quantities) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    size_frame = frame.loc[frame["N"] == size]
    prepared: dict[str, tuple[pd.DataFrame, list[int]]] = {}
    for model in ["NNP", "mW"]:
        model_frame = size_frame.loc[size_frame["model"] == model]
        if not model_frame.empty:
            prepared[model] = concatenate_runs(model_frame)

    for ax, (column, ylabel) in zip(axes_flat, quantities):
        for model, (model_frame, boundaries) in prepared.items():
            valid = model_frame[column].notna()
            ax.plot(
                model_frame.loc[valid, "configuration_index"],
                model_frame.loc[valid, column],
                linewidth=0.65,
                alpha=0.85,
                color=COLORS[model],
                label=model,
                rasterized=True,
            )
            if model == "NNP":
                for boundary in boundaries:
                    ax.axvline(boundary - 0.5, color="0.65", linewidth=0.6, alpha=0.6)
        ax.set_xlabel("Concatenated configuration index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

    if prepared:
        axes_flat[0].legend(frameon=False)
    for ax in axes_flat[len(quantities):]:
        ax.set_visible(False)
    fig.suptitle(f"Thermodynamic traces, N = {size}", y=0.998)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"thermo_vs_configuration_N_{size}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


def main() -> int:
    """Generate one thermodynamic trace figure for every system size."""
    args = parse_args()
    try:
        frame, quantities = load_and_prepare(args.input.resolve())
        output_dir = args.output_dir.resolve()
        for size in sorted(frame["N"].unique()):
            plot_system_size(frame, int(size), quantities, output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Plotted {len(quantities)} quantities after removing two endpoints per run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
