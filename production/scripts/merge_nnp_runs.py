#!/usr/bin/env python3
"""Concatenate independent NNP runs by system size without averaging frames."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["model", "N", "run"]
DERIVED_COLUMNS = [
    "delta_U",
    "delta_u",
    "poteng_per_particle",
    "cross_poteng_per_particle",
]
SUMMARY_COLUMNS = [
    "model",
    "N",
    "run",
    "n_frames",
    "n_original_runs",
    "mean_poteng_per_particle",
    "mean_cross_poteng_per_particle",
    "mean_delta_u",
    "std_delta_u",
]
REQUIRED_COLUMNS = {"model", "N", "run", "step", "poteng", "cross_poteng"}
CSV_FLOAT_FORMAT = "%.10g"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[1]
    analysis_dir = project_root / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=analysis_dir / "thermo_all_with_cross.csv",
        help="Input cross-energy CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "thermo_all_with_cross_merged.csv",
        help="Merged output CSV",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=analysis_dir / "merged_runs_summary.csv",
        help="Merged-run summary CSV",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing merged output CSV (refused by default)",
    )
    return parser.parse_args()


def normalize_integer_column(series: pd.Series, label: str) -> pd.Series:
    """Convert an integer-valued numeric column without silent truncation."""
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(f"{label} contains missing or non-integer values")
    return numeric.astype("int64")


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the master table and validate columns needed for safe merging."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    frame["model"] = frame["model"].astype(str)
    unknown_models = sorted(set(frame["model"]) - {"mW", "NNP"})
    if unknown_models:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown_models)}")

    frame["N"] = normalize_integer_column(frame["N"], "N")
    frame["run"] = normalize_integer_column(frame["run"], "run")
    frame["step"] = normalize_integer_column(frame["step"], "step")
    if (frame["N"] <= 0).any():
        raise ValueError("N must be positive")

    for column in ["poteng", "cross_poteng"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        finite = np.isfinite(frame[column].to_numpy(dtype=float))
        if not finite.all():
            raise ValueError(f"{column} contains {int((~finite).sum()):,} non-finite values")

    key_columns = ["model", "N", "run", "step"]
    duplicate = frame.duplicated(key_columns, keep=False)
    if duplicate.any():
        preview = frame.loc[duplicate, key_columns].drop_duplicates().head(10)
        raise ValueError(
            "Input has duplicate (model, N, run, step) keys. First duplicates:\n"
            + preview.to_string(index=False)
        )

    bad_mw_runs = sorted(frame.loc[(frame["model"] == "mW") & (frame["run"] != 1), "run"].unique())
    if bad_mw_runs:
        warnings.warn(f"mW contains run values other than 1: {bad_mw_runs}")
    return frame


def insert_after(frame: pd.DataFrame, after: str, name: str, values: pd.Series) -> None:
    """Insert a column at a readable position in the existing schema."""
    if name in frame.columns:
        frame.drop(columns=name, inplace=True)
    position = frame.columns.get_loc(after) + 1
    frame.insert(position, name, values)


def merge_nnp_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Concatenate NNP runs at fixed N and assign final-group sample IDs."""
    merged = frame.copy()

    # Preserve provenance before replacing the NNP run label with the merged
    # ensemble convention. mW source_run remains identical to its run value.
    original_run = merged["run"].copy()
    insert_after(merged, "run", "source_run", original_run)
    merged.loc[merged["model"] == "NNP", "run"] = 0

    # Stable sorting makes concatenation deterministic: original run order is
    # retained within each NNP system size, followed by saved-step order.
    merged = merged.sort_values(
        ["model", "N", "run", "source_run", "step"], kind="stable"
    ).reset_index(drop=True)
    sample_id = merged.groupby(GROUP_COLUMNS, sort=False).cumcount().astype("int64")
    insert_after(merged, "source_run", "sample_id", sample_id)

    # Recompute rather than trusting potentially stale derived input columns.
    merged["delta_U"] = merged["cross_poteng"] - merged["poteng"]
    merged["delta_u"] = merged["delta_U"] / merged["N"]
    merged["poteng_per_particle"] = merged["poteng"] / merged["N"]
    merged["cross_poteng_per_particle"] = merged["cross_poteng"] / merged["N"]
    return merged


def validate_merged(original: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Check row preservation, provenance, separation, and sample-ID uniqueness."""
    if len(merged) != len(original):
        raise ValueError(f"Row count changed from {len(original):,} to {len(merged):,}")
    if not (merged.loc[merged["model"] == "NNP", "run"] == 0).all():
        raise ValueError("Some NNP rows were not assigned run=0")
    if not (merged.loc[merged["model"] == "mW", "run"] == 1).all():
        raise ValueError("Some mW rows no longer have run=1")

    provenance_key = ["model", "N", "source_run", "step"]
    if merged.duplicated(provenance_key).any():
        raise ValueError("Merged data lost unique original-run provenance")
    sample_key = GROUP_COLUMNS + ["sample_id"]
    if merged.duplicated(sample_key).any():
        raise ValueError("sample_id is not unique within final groups")

    # Every final group's IDs must be exactly 0, ..., n_frames - 1.
    for key, group in merged.groupby(GROUP_COLUMNS, sort=True):
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(group["sample_id"].to_numpy(), expected):
            raise ValueError(f"Non-contiguous sample_id values in group {key}")


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one descriptive row per final trajectory/merged ensemble."""
    records: list[dict[str, object]] = []
    for (model, size, run), group in frame.groupby(GROUP_COLUMNS, sort=True):
        delta_u = group["delta_u"].to_numpy(dtype=float)
        records.append(
            {
                "model": model,
                "N": int(size),
                "run": int(run),
                "n_frames": len(group),
                "n_original_runs": int(group["source_run"].nunique()),
                "mean_poteng_per_particle": float(group["poteng_per_particle"].mean()),
                "mean_cross_poteng_per_particle": float(
                    group["cross_poteng_per_particle"].mean()
                ),
                "mean_delta_u": float(np.mean(delta_u)),
                "std_delta_u": float(np.std(delta_u, ddof=0)),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def write_outputs(
    merged: pd.DataFrame, summary: pd.DataFrame, output: Path, summary_path: Path
) -> None:
    """Write CSV outputs and Parquet when the installed stack supports it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved {len(merged):,} frames to {output}")

    if importlib.util.find_spec("pyarrow") is not None:
        parquet_path = output.with_suffix(".parquet")
        try:
            merged.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f"Saved {len(merged):,} frames to {parquet_path}")
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            warnings.warn(f"Could not write Parquet output: {exc}")
    else:
        print("pyarrow is not installed; skipping Parquet output.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved summary to {summary_path}")


def print_nnp_summary(summary: pd.DataFrame) -> None:
    """Print NNP frame and source-run counts after concatenation."""
    nnp = summary.loc[
        summary["model"] == "NNP", ["N", "run", "n_frames", "n_original_runs"]
    ]
    print("\nMerged NNP ensembles:")
    if nnp.empty:
        print("No NNP rows found.")
    else:
        print(nnp.to_string(index=False))


def main() -> int:
    """Run the NNP concatenation workflow."""
    args = parse_args()
    try:
        # The merged table is the master dataset used by every analysis, so
        # never replace it silently.
        if args.output.resolve().exists() and not args.overwrite:
            raise ValueError(
                f"{args.output.resolve()} already exists; pass --overwrite to "
                "replace it or --output to write elsewhere"
            )
        original = load_and_validate(args.input.resolve())
        merged = merge_nnp_runs(original)
        validate_merged(original, merged)
        summary = make_summary(merged)
        write_outputs(merged, summary, args.output.resolve(), args.summary.resolve())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_nnp_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
