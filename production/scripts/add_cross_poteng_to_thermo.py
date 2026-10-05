#!/usr/bin/env python3
"""Add cross-Hamiltonian potential energies to the retained thermo dataset."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["model", "N", "run", "step"]
CSV_FLOAT_FORMAT = "%.10g"
DERIVED_COLUMNS = [
    "cross_poteng",
    "delta_U",
    "delta_u",
    "poteng_per_particle",
    "cross_poteng_per_particle",
]
OUTPUT_COLUMNS = [
    "model",
    "N",
    "run",
    "step",
    "temperature",
    "poteng",
    "cross_poteng",
    "delta_u",
    "toteng",
    "press",
    "volume",
    "density",
]
SUMMARY_COLUMNS = [
    "model",
    "N",
    "run",
    "n_frames",
    "n_cross_matched",
    "mean_poteng",
    "mean_cross_poteng",
    "mean_delta_U",
    "std_delta_U",
    "mean_delta_u",
    "std_delta_u",
]
NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?$"
    r"|^[+-]?(?:inf|nan)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        default=project_root / "analysis" / "thermo_all.csv",
        help="Master thermo CSV (default: PROJECT_ROOT/analysis/thermo_all.csv)",
    )
    parser.add_argument(
        "--mw-rerun-root",
        type=Path,
        default=project_root / "results/full_mw_trajectory",
        help="NNP evaluations of every saved mW frame, as N_<N>/log.rerun_NNP.lammps "
        "(default: results/full_mw_trajectory)",
    )
    parser.add_argument(
        "--corrected-mw-root",
        type=Path,
        default=project_root / "results/corrected_mw_rerun",
        help="Corrected mW evaluations of NNP frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV when not overwriting (default: analysis/thermo_all_with_cross.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary CSV (default: analysis/cross_poteng_summary.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Back up and overwrite the master CSV instead of writing a new CSV",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write output even when some cross_poteng values are missing",
    )
    return parser.parse_args()


def parse_lammps_log(path: Path) -> pd.DataFrame:
    """Extract Step and potential energy from every thermo block in a log."""
    records: list[dict[str, float]] = []
    header: list[str] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0].lower() == "step":
                header = [field.lower() for field in fields]
                if "poteng" not in header and "pe" not in header:
                    header = None
                continue

            if header is None or len(fields) != len(header):
                continue
            if not all(NUMBER_RE.fullmatch(value) for value in fields):
                continue

            values = dict(zip(header, (float(value) for value in fields)))
            energy_name = "poteng" if "poteng" in values else "pe"
            records.append(
                {"step": values["step"], "cross_poteng": values[energy_name]}
            )

    return pd.DataFrame.from_records(records, columns=["step", "cross_poteng"])


def expected_rerun_logs(
    master_df: pd.DataFrame, mw_rerun_root: Path, corrected_mw_root: Path
) -> list[tuple[str, int, int, Path]]:
    """Build the rerun-log path for every model/size/run in the master table."""
    jobs: list[tuple[str, int, int, Path]] = []
    groups = master_df[["model", "N", "run"]].drop_duplicates()
    for model, size, run in groups.itertuples(index=False, name=None):
        if model == "mW":
            # Full-trajectory rerun; the older lammps/mW_NVT/N_*/ rerun log
            # covered only the first 2,001 saved frames.
            path = mw_rerun_root / f"N_{size}" / "log.rerun_NNP.lammps"
        elif model == "NNP":
            path = (
                corrected_mw_root
                / f"N_{size}"
                / f"run_{run}"
                / "log.rerun_mW.lammps"
            )
        else:
            warnings.warn(
                f"No rerun path convention is defined for model={model!r}, "
                f"N={size}, run={run}"
            )
            continue
        jobs.append((str(model), int(size), int(run), path))
    return jobs


def parse_all_rerun_logs(
    mw_rerun_root: Path, corrected_mw_root: Path, master_df: pd.DataFrame
) -> pd.DataFrame:
    """Parse all rerun logs required by the groups in the master dataset."""
    frames: list[pd.DataFrame] = []
    for model, size, run, path in expected_rerun_logs(master_df, mw_rerun_root, corrected_mw_root):
        if not path.is_file():
            warnings.warn(
                f"Missing rerun log for model={model}, N={size}, run={run}: {path}"
            )
            continue
        try:
            log_text = path.read_text(encoding="utf-8", errors="replace")
            # Replicated mW systems need the full atom count in their rerun.
            atom_counts = [int(value) for value in re.findall(
                r"^\s*(\d+)\s+atoms\s*$", log_text, flags=re.MULTILINE
            )]
            if atom_counts != [size]:
                raise ValueError(
                    f"Rerun atom count {atom_counts} does not match N={size}: {path}"
                )
            if "Total wall time:" not in log_text or re.search(
                r"^ERROR(?: on proc \d+)?:", log_text, flags=re.MULTILINE
            ):
                raise ValueError(f"Rerun did not finish cleanly: {path}")
            frame = parse_lammps_log(path)
        except OSError as exc:
            warnings.warn(f"Could not read rerun log {path}: {exc}")
            continue
        if frame.empty:
            warnings.warn(f"No readable Step/PotEng thermo rows found in {path}")
            continue
        frame.insert(0, "run", run)
        frame.insert(0, "N", size)
        frame.insert(0, "model", model)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + ["cross_poteng"])
    rerun_df = pd.concat(frames, ignore_index=True)
    rerun_df["step"] = normalize_integer_column(rerun_df["step"], "rerun step")
    return rerun_df


def normalize_integer_column(series: pd.Series, label: str) -> pd.Series:
    """Convert an integer-valued numeric column to int64 without truncation."""
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(f"{label} contains missing or non-integer values")
    return numeric.astype("int64")


def load_master(path: Path) -> pd.DataFrame:
    """Read and normalize the master table while retaining its full schema."""
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["poteng"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    frame["model"] = frame["model"].astype(str)
    frame["N"] = normalize_integer_column(frame["N"], "master N")
    frame["run"] = normalize_integer_column(frame["run"], "master run")
    frame["step"] = normalize_integer_column(frame["step"], "master step")
    frame["poteng"] = pd.to_numeric(frame["poteng"], errors="raise")
    return frame


def check_unique_keys(frame: pd.DataFrame, label: str) -> None:
    """Raise with a useful preview if a logical key is duplicated."""
    duplicate = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicate.any():
        preview = frame.loc[duplicate, KEY_COLUMNS].drop_duplicates().head(10)
        raise ValueError(
            f"{label} has duplicate (model, N, run, step) keys. First duplicates:\n"
            f"{preview.to_string(index=False)}"
        )


def warn_step_differences(master_df: pd.DataFrame, rerun_df: pd.DataFrame) -> None:
    """Warn separately about master-only and rerun-only steps in every run."""
    group_columns = ["model", "N", "run"]
    master_groups = {
        key: set(group["step"].tolist())
        for key, group in master_df.groupby(group_columns, sort=True)
    }
    rerun_groups = {
        key: set(group["step"].tolist())
        for key, group in rerun_df.groupby(group_columns, sort=True)
    }

    for key in sorted(set(master_groups) | set(rerun_groups)):
        master_steps = master_groups.get(key, set())
        rerun_steps = rerun_groups.get(key, set())
        missing = master_steps - rerun_steps
        extra = rerun_steps - master_steps
        label = f"model={key[0]}, N={key[1]}, run={key[2]}"
        if missing:
            preview = ", ".join(str(step) for step in sorted(missing)[:10])
            suffix = " ..." if len(missing) > 10 else ""
            warnings.warn(
                f"{label}: {len(missing)} master step(s) absent from rerun data: "
                f"{preview}{suffix}"
            )
        if extra:
            preview = ", ".join(str(step) for step in sorted(extra)[:10])
            suffix = " ..." if len(extra) > 10 else ""
            warnings.warn(
                f"{label}: {len(extra)} rerun step(s) absent from master data: "
                f"{preview}{suffix}"
            )


def add_cross_poteng(master_df: pd.DataFrame, rerun_df: pd.DataFrame) -> pd.DataFrame:
    """Add cross energy by a validated logical-key merge, preserving row order."""
    # Re-running the tool should replace only columns that this tool generated.
    base = master_df.drop(columns=DERIVED_COLUMNS, errors="ignore").copy()
    base["_master_row_order"] = np.arange(len(base), dtype=np.int64)
    merged = base.merge(
        rerun_df[KEY_COLUMNS + ["cross_poteng"]],
        on=KEY_COLUMNS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    merged = merged.sort_values("_master_row_order", kind="stable")
    merged = merged.drop(columns="_master_row_order").reset_index(drop=True)

    merged["delta_U"] = merged["cross_poteng"] - merged["poteng"]
    merged["delta_u"] = merged["delta_U"] / merged["N"]
    merged["poteng_per_particle"] = merged["poteng"] / merged["N"]
    merged["cross_poteng_per_particle"] = merged["cross_poteng"] / merged["N"]
    return merged


def validate_matches(
    master_df: pd.DataFrame,
    rerun_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    allow_missing: bool,
) -> None:
    """Validate uniqueness, row preservation, and complete finite matching."""
    check_unique_keys(master_df, "Master dataset")
    check_unique_keys(rerun_df, "Parsed rerun data")
    warn_step_differences(master_df, rerun_df)

    if len(merged_df) != len(master_df):
        raise ValueError(
            f"Merge changed row count from {len(master_df):,} to {len(merged_df):,}"
        )
    check_unique_keys(merged_df, "Merged dataset")

    finite_cross = np.isfinite(merged_df["cross_poteng"].to_numpy(dtype=float))
    n_missing = int((~finite_cross).sum())
    if n_missing:
        message = f"Final table has {n_missing:,} missing or non-finite cross_poteng values"
        if not allow_missing:
            raise ValueError(message + "; rerun with --allow-missing to write partial output")
        warnings.warn(message)


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize original, cross, and difference energies by simulation."""
    summary = (
        frame.groupby(["model", "N", "run"], sort=True)
        .agg(
            n_frames=("step", "size"),
            n_cross_matched=("cross_poteng", "count"),
            mean_poteng=("poteng", "mean"),
            mean_cross_poteng=("cross_poteng", "mean"),
            mean_delta_U=("delta_U", "mean"),
            std_delta_U=("delta_U", "std"),
            mean_delta_u=("delta_u", "mean"),
            std_delta_u=("delta_u", "std"),
        )
        .reset_index()
    )
    return summary[SUMMARY_COLUMNS]


def write_outputs(
    merged_df: pd.DataFrame, summary_df: pd.DataFrame, args: argparse.Namespace
) -> tuple[Path, Path | None, Path]:
    """Write CSV, optional Parquet, and summary outputs safely."""
    master_path = args.master.resolve()
    analysis_dir = master_path.parent
    if args.overwrite:
        output_path = master_path
        backup_path = master_path.with_suffix(master_path.suffix + ".bak")
        shutil.copy2(master_path, backup_path)
        print(f"Backed up master dataset to {backup_path}")
    else:
        output_path = (
            args.output.resolve()
            if args.output
            else analysis_dir / "thermo_all_with_cross.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the main table compact; derived differences remain available in the
    # separately written grouped summary.
    output_df = merged_df[OUTPUT_COLUMNS]
    output_df.to_csv(output_path, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved {len(output_df):,} rows to {output_path}")

    parquet_path: Path | None = None
    if importlib.util.find_spec("pyarrow") is not None:
        parquet_path = output_path.with_suffix(".parquet")
        try:
            output_df.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f"Saved {len(output_df):,} rows to {parquet_path}")
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            parquet_path = None
            warnings.warn(f"Could not write Parquet output: {exc}")
    else:
        print("pyarrow is not installed; skipping Parquet output.")

    summary_path = (
        args.summary.resolve()
        if args.summary
        else analysis_dir / "cross_poteng_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved summary to {summary_path}")
    return output_path, parquet_path, summary_path


def main() -> int:
    """Run the cross-energy augmentation workflow."""
    args = parse_args()
    try:
        master_df = load_master(args.master.resolve())
        check_unique_keys(master_df, "Master dataset")
        rerun_df = parse_all_rerun_logs(
            args.mw_rerun_root.resolve(), args.corrected_mw_root.resolve(), master_df
        )
        check_unique_keys(rerun_df, "Parsed rerun data")
        merged_df = add_cross_poteng(master_df, rerun_df)
        validate_matches(master_df, rerun_df, merged_df, args.allow_missing)
        summary_df = make_summary(merged_df)
        write_outputs(merged_df, summary_df, args)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nCross-potential-energy summary:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
