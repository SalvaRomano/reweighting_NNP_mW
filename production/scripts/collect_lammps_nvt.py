#!/usr/bin/env python3
"""Collect thermo output from mW and NNP LAMMPS NVT simulations."""

from __future__ import annotations

import argparse
import importlib.util
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "model",
    "N",
    "run",
    "step",
    "temperature",
    "poteng",
    "toteng",
    "press",
    "volume",
    "density",
    "source_file",
]

# LAMMPS column names are case-sensitive in the log, but matching them without
# case sensitivity makes the parser tolerant of output from different versions.
THERMO_NAMES = {
    "step": "step",
    "temp": "temperature",
    "poteng": "poteng",
    "pe": "poteng",
    "toteng": "toteng",
    "etotal": "toteng",
    "press": "press",
    "volume": "volume",
    "vol": "volume",
    "density": "density",
}
NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?$"
    r"|^[+-]?(?:inf|nan)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing mW_NVT and NN_NVT (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT_ROOT/analysis)",
    )
    return parser.parse_args()


def discover_runs(project_root: Path) -> list[tuple[str, int, int, Path]]:
    """Return model, system size, run number, and log path for expected runs."""
    runs: list[tuple[str, int, int, Path]] = []

    for size_dir in sorted((project_root / "mW_NVT").glob("N_*")):
        match = re.fullmatch(r"N_(\d+)", size_dir.name)
        if match and size_dir.is_dir():
            runs.append(("mW", int(match.group(1)), 1, size_dir / "log.lammps"))

    for size_dir in sorted((project_root / "NN_NVT").glob("N_*")):
        size_match = re.fullmatch(r"N_(\d+)", size_dir.name)
        if not size_match or not size_dir.is_dir():
            continue
        for run_dir in sorted(size_dir.glob("run_*")):
            run_match = re.fullmatch(r"run_(\d+)", run_dir.name)
            if run_match and run_dir.is_dir():
                runs.append(
                    ("NNP", int(size_match.group(1)), int(run_match.group(1)),
                     run_dir / "log.lammps")
                )

    return runs


def parse_thermo_blocks(log_text: str) -> list[dict[str, float]]:
    """Extract numeric rows from all recognized thermo blocks in a LAMMPS log."""
    records: list[dict[str, float]] = []
    header: list[str] | None = None

    for line in log_text.splitlines():
        fields = line.split()
        lowered = [field.lower() for field in fields]

        # A thermo header must contain Step plus at least one supported property.
        if "step" in lowered and any(name in THERMO_NAMES for name in lowered if name != "step"):
            header = lowered
            continue

        if header is None or len(fields) != len(header):
            continue
        if not all(NUMBER_RE.fullmatch(value) for value in fields):
            continue

        values = [float(value) for value in fields]
        record = {column: np.nan for column in OUTPUT_COLUMNS[3:10]}
        for lammps_name, value in zip(header, values):
            standard_name = THERMO_NAMES.get(lammps_name)
            if standard_name is not None:
                record[standard_name] = value
        records.append(record)

    return records


def classify_termination(log_text: str) -> str:
    """Classify how LAMMPS stopped based on its final termination message."""
    lost_atoms = list(
        re.finditer(r"^ERROR(?: on proc \d+)?:\s+Lost atoms\b", log_text, re.MULTILINE)
    )
    other_errors = list(
        re.finditer(r"^ERROR(?: on proc \d+)?:", log_text, re.MULTILINE)
    )
    clean_exit_position = log_text.rfind("Total wall time:")
    last_error_position = other_errors[-1].start() if other_errors else -1

    # Logs can be appended or contain more than one LAMMPS invocation, so the
    # latest termination marker determines the status of the collected run.
    if clean_exit_position > last_error_position:
        return "complete"
    if lost_atoms and lost_atoms[-1].start() == last_error_position:
        return "lost_atoms"
    if other_errors:
        return "other_error"
    # Batch schedulers often terminate a job without adding anything to the log.
    return "interrupted"


def collect(project_root: Path) -> pd.DataFrame:
    """Collect complete simulations into one consistently shaped DataFrame."""
    all_records: list[dict[str, object]] = []
    discovered = discover_runs(project_root)
    if not discovered:
        warnings.warn(f"No simulation directories found below {project_root}")

    for model, size, run, log_path in discovered:
        if not log_path.is_file():
            warnings.warn(f"Skipping missing log: {log_path}")
            continue
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.warn(f"Skipping unreadable log {log_path}: {exc}")
            continue

        termination = classify_termination(log_text)
        if termination == "other_error":
            warnings.warn(f"Skipping run containing an unrecognized LAMMPS error: {log_path}")
            continue

        thermo_records = parse_thermo_blocks(log_text)
        if termination in {"lost_atoms", "interrupted"} and thermo_records:
            # The last printed state may coincide with the unstable step or with
            # a partially terminated write, so omit it from non-clean runs.
            thermo_records.pop()
            reason = "lost atoms" if termination == "lost_atoms" else "likely time limit"
            warnings.warn(
                f"Including {reason} run after removing its last thermo row: {log_path}"
            )
        if not thermo_records:
            warnings.warn(f"Skipping run with no readable thermo rows: {log_path}")
            continue

        try:
            source_file = str(log_path.relative_to(project_root))
        except ValueError:
            source_file = str(log_path.resolve())
        for record in thermo_records:
            record.update(
                {"model": model, "N": size, "run": run, "source_file": source_file}
            )
            all_records.append(record)

    frame = pd.DataFrame.from_records(all_records, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame["N"] = frame["N"].astype("int64")
        frame["run"] = frame["run"].astype("int64")
        # Step is normally integral; nullable Int64 also handles a malformed or
        # absent Step field gracefully without changing the public schema.
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce").astype("Int64")
        frame = frame.sort_values(["model", "N", "run", "step"], kind="stable")
        frame = frame.reset_index(drop=True)
    return frame


def save_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "thermo_all.csv"
    frame.to_csv(csv_path, index=False)
    print(f"Saved {len(frame):,} frames to {csv_path}")

    if importlib.util.find_spec("pyarrow") is not None:
        parquet_path = output_dir / "thermo_all.parquet"
        try:
            frame.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f"Saved {len(frame):,} frames to {parquet_path}")
        except (ImportError, OSError, ValueError) as exc:
            warnings.warn(f"Could not write Parquet output: {exc}")
    else:
        print("pyarrow is not installed; skipping Parquet output.")


def print_summary(frame: pd.DataFrame) -> None:
    print("\nFrames per simulation:")
    if frame.empty:
        print("No complete simulations with readable thermo data were found.")
        return
    summary = (
        frame.groupby(["model", "N", "run"], sort=True)
        .size()
        .rename("frames")
        .reset_index()
    )
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "analysis"
    frame = collect(project_root)
    save_outputs(frame, output_dir)
    print_summary(frame)


if __name__ == "__main__":
    main()
