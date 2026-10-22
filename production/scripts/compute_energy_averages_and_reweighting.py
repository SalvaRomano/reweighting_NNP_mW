#!/usr/bin/env python3
"""Compute direct and bidirectionally reweighted energies at each system size."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


KB_EV_PER_K = 8.617333262145e-5
CSV_FLOAT_FORMAT = "%.10g"
REQUIRED_COLUMNS = {
    "model", "N", "source_run", "sample_id", "step", "poteng", "cross_poteng"
}
OUTPUT_COLUMNS = [
    "N", "n_frames_NNP", "n_frames_mW", "U_NNP_in_NNP", "U_mW_in_mW",
    "U_mW_in_NNP", "U_NNP_in_mW",
    "U_mW_reweighted_from_NNP", "U_NNP_reweighted_from_mW",
    "error_before_NNP_to_mW", "abs_error_before_NNP_to_mW",
    "error_after_NNP_to_mW", "abs_error_after_NNP_to_mW",
    "error_before_mW_to_NNP", "abs_error_before_mW_to_NNP",
    "error_after_mW_to_NNP", "abs_error_after_mW_to_NNP",
    "error_NNP_to_mW", "error_mW_to_NNP", "neff_NNP_to_mW",
    "neff_fraction_NNP_to_mW",
    "max_weight_fraction_NNP_to_mW", "max_weight_sample_id_NNP_to_mW",
    "top10_weight_fraction_NNP_to_mW", "top10_weight_sample_ids_NNP_to_mW",
    "neff_mW_to_NNP", "neff_fraction_mW_to_NNP",
    "max_weight_fraction_mW_to_NNP", "max_weight_sample_id_mW_to_NNP",
    "top10_weight_fraction_mW_to_NNP", "top10_weight_sample_ids_mW_to_NNP",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    analysis_dir = Path(__file__).resolve().parents[1] / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=analysis_dir / "thermo_all_with_cross_merged.csv")
    parser.add_argument("--output", type=Path,
                        default=analysis_dir / "energy_averages_and_reweighting.csv")
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Single temperature in K used only for beta (default: 300)",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the merged dataset and strictly validate energies and sizes."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame["model"] = frame["model"].astype(str)
    unknown = sorted(set(frame["model"]) - {"NNP", "mW"})
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}")

    size = pd.to_numeric(frame["N"], errors="raise")
    if (size.isna().any() or not np.isfinite(size).all()
            or not np.all(np.equal(size, np.floor(size))) or (size <= 0).any()):
        raise ValueError("N must contain finite positive integers")
    frame["N"] = size.astype("int64")
    for column in ["source_run", "sample_id", "step"]:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{column} must contain finite integer values")
        frame[column] = numeric.astype("int64")
    for column in ["poteng", "cross_poteng"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["cross_poteng"].isna().any():
        raise ValueError(
            f"cross_poteng contains {int(frame['cross_poteng'].isna().sum()):,} missing values"
        )
    for column in ["poteng", "cross_poteng"]:
        finite = np.isfinite(frame[column].to_numpy(dtype=float))
        if not finite.all():
            raise ValueError(f"{column} contains {int((~finite).sum()):,} non-finite values")

    # Always reconstruct target-minus-source energy from the primary columns.
    frame["delta_U"] = frame["cross_poteng"] - frame["poteng"]
    return frame


def remove_trajectory_endpoints(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove first and last saved steps independently from every trajectory."""
    trajectory_columns = ["model", "N", "source_run"]
    counts = frame.groupby(trajectory_columns, sort=True).size()
    too_short = counts[counts < 3]
    if not too_short.empty:
        raise ValueError(
            "Cannot remove two endpoints from trajectories with fewer than three frames:\n"
            + too_short.to_string()
        )

    grouped_steps = frame.groupby(trajectory_columns, sort=False)["step"]
    first_step = grouped_steps.transform("min")
    last_step = grouped_steps.transform("max")
    keep = (frame["step"] != first_step) & (frame["step"] != last_step)
    trimmed = frame.loc[keep].copy()

    # Record and validate the removal count for each original NNP run and mW
    # trajectory before the NNP samples are pooled by system size.
    before = counts.rename("n_before")
    after = trimmed.groupby(trajectory_columns, sort=True).size().rename("n_after")
    removal = pd.concat([before, after], axis=1).fillna(0).astype("int64").reset_index()
    removal["n_removed"] = removal["n_before"] - removal["n_after"]
    if not (removal["n_removed"] == 2).all():
        raise ValueError("Endpoint trimming did not remove exactly two frames per trajectory")
    return trimmed, removal


def direct_average(group: pd.DataFrame) -> float:
    """Average source-Hamiltonian energy per particle in its own ensemble."""
    if group.empty:
        return np.nan
    return float(np.mean(group["poteng"] / group["N"]))


def cross_average(group: pd.DataFrame) -> float:
    """Average cross-Hamiltonian energy per particle without reweighting."""
    if group.empty:
        return np.nan
    return float(np.mean(group["cross_poteng"] / group["N"]))


def reweighted_average(group: pd.DataFrame, beta: float) -> dict[str, object]:
    """Reweight source frames to the cross Hamiltonian using shifted weights."""
    if group.empty:
        return {
            "energy": np.nan,
            "neff": np.nan,
            "neff_fraction": np.nan,
            "max_weight_fraction": np.nan,
            "max_weight_sample_id": np.nan,
            "top10_weight_fraction": np.nan,
            "top10_weight_sample_ids": "",
        }
    logw = -beta * group["delta_U"].to_numpy(dtype=float)
    target_u = (group["cross_poteng"] / group["N"]).to_numpy(dtype=float)

    # Never exponentiate unshifted log-weights.
    weights = np.exp(logw - np.max(logw))
    weight_sum = float(np.sum(weights))
    weight_square_sum = float(np.sum(weights * weights))
    if weight_sum <= 0.0 or weight_square_sum <= 0.0:
        raise ValueError("Shifted weights are numerically degenerate")
    probabilities = weights / weight_sum
    # Stable sorting makes the reported configuration IDs reproducible if two
    # configurations happen to have equal normalized weights.
    ranked = np.argsort(-probabilities, kind="stable")
    top = ranked[: min(10, len(ranked))]
    sample_ids = group["sample_id"].to_numpy(dtype=np.int64)
    neff = weight_sum * weight_sum / weight_square_sum
    return {
        "energy": float(np.sum(weights * target_u) / weight_sum),
        "neff": neff,
        "neff_fraction": neff / len(group),
        "max_weight_fraction": float(probabilities[ranked[0]]),
        "max_weight_sample_id": int(sample_ids[ranked[0]]),
        "top10_weight_fraction": float(np.sum(probabilities[top])),
        "top10_weight_sample_ids": ";".join(str(value) for value in sample_ids[top]),
    }


def compute_for_size(frame: pd.DataFrame, size: int, beta: float) -> dict[str, object]:
    """Compute two direct and two reweighted estimates for one system size."""
    rows = frame.loc[frame["N"] == size]
    nnp = rows.loc[rows["model"] == "NNP"]
    mw = rows.loc[rows["model"] == "mW"]
    if nnp.empty:
        warnings.warn(f"N={size}: missing NNP ensemble")
    if mw.empty:
        warnings.warn(f"N={size}: missing mW ensemble")

    U_NNP_in_NNP = direct_average(nnp)
    U_mW_in_mW = direct_average(mw)
    U_mW_in_NNP = cross_average(nnp)
    U_NNP_in_mW = cross_average(mw)
    nnp_to_mw = reweighted_average(nnp, beta)
    mw_to_nnp = reweighted_average(mw, beta)
    # Every average above is already in eV per particle; differences need no
    # further division by system size.
    error_before_nnp_to_mw = U_mW_in_NNP - U_mW_in_mW
    error_before_mw_to_nnp = U_NNP_in_mW - U_NNP_in_NNP
    error_after_nnp_to_mw = nnp_to_mw["energy"] - U_mW_in_mW
    error_after_mw_to_nnp = mw_to_nnp["energy"] - U_NNP_in_NNP
    return {
        "N": size, "n_frames_NNP": len(nnp), "n_frames_mW": len(mw),
        "U_NNP_in_NNP": U_NNP_in_NNP, "U_mW_in_mW": U_mW_in_mW,
        "U_mW_in_NNP": U_mW_in_NNP, "U_NNP_in_mW": U_NNP_in_mW,
        "U_mW_reweighted_from_NNP": nnp_to_mw["energy"],
        "U_NNP_reweighted_from_mW": mw_to_nnp["energy"],
        "error_before_NNP_to_mW": error_before_nnp_to_mw,
        "abs_error_before_NNP_to_mW": abs(error_before_nnp_to_mw),
        "error_after_NNP_to_mW": error_after_nnp_to_mw,
        "abs_error_after_NNP_to_mW": abs(error_after_nnp_to_mw),
        "error_before_mW_to_NNP": error_before_mw_to_nnp,
        "abs_error_before_mW_to_NNP": abs(error_before_mw_to_nnp),
        "error_after_mW_to_NNP": error_after_mw_to_nnp,
        "abs_error_after_mW_to_NNP": abs(error_after_mw_to_nnp),
        "error_NNP_to_mW": error_after_nnp_to_mw,
        "error_mW_to_NNP": error_after_mw_to_nnp,
        "neff_NNP_to_mW": nnp_to_mw["neff"],
        "neff_fraction_NNP_to_mW": nnp_to_mw["neff_fraction"],
        "max_weight_fraction_NNP_to_mW": nnp_to_mw["max_weight_fraction"],
        "max_weight_sample_id_NNP_to_mW": nnp_to_mw["max_weight_sample_id"],
        "top10_weight_fraction_NNP_to_mW": nnp_to_mw["top10_weight_fraction"],
        "top10_weight_sample_ids_NNP_to_mW": nnp_to_mw["top10_weight_sample_ids"],
        "neff_mW_to_NNP": mw_to_nnp["neff"],
        "neff_fraction_mW_to_NNP": mw_to_nnp["neff_fraction"],
        "max_weight_fraction_mW_to_NNP": mw_to_nnp["max_weight_fraction"],
        "max_weight_sample_id_mW_to_NNP": mw_to_nnp["max_weight_sample_id"],
        "top10_weight_fraction_mW_to_NNP": mw_to_nnp["top10_weight_fraction"],
        "top10_weight_sample_ids_mW_to_NNP": mw_to_nnp["top10_weight_sample_ids"],
    }


def compute_summary(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Compute one row per N, with no temperature-based grouping."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("--temperature must be a finite positive value")
    beta = 1.0 / (KB_EV_PER_K * temperature)
    records = [compute_for_size(frame, int(n), beta) for n in sorted(frame["N"].unique())]
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def write_outputs(summary: pd.DataFrame, output: Path) -> None:
    """Write the compact CSV and optional Parquet table."""
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved {len(summary):,} system sizes to {output}")
    if importlib.util.find_spec("pyarrow") is not None:
        try:
            summary.to_parquet(output.with_suffix(".parquet"), index=False, engine="pyarrow")
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            warnings.warn(f"Could not write Parquet output: {exc}")


def main() -> int:
    """Run the direct and reweighted energy analysis."""
    args = parse_args()
    try:
        frame = load_and_validate(args.input.resolve())
        trimmed, removal = remove_trajectory_endpoints(frame)
        summary = compute_summary(trimmed, args.temperature)
        write_outputs(summary, args.output.resolve())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Reweighting beta used T = {args.temperature:g} K (not used for grouping).")
    print(f"Removed {int(removal['n_removed'].sum()):,} endpoint frames "
          f"from {len(removal):,} original trajectories.")
    print("\nDirect and reweighted energy per particle:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
