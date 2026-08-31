#!/usr/bin/env python3
"""Collect original and cross-Hamiltonian energies aligned strictly by Step."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SIZE_RE = re.compile(r"N_(\d+)$")
RUN_RE = re.compile(r"run_(\d+)$")
NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?$"
    r"|^[+-]?(?:inf|nan)$",
    re.IGNORECASE,
)
THERMO_COLUMNS = {
    "step": "step",
    "time": "time",
    "temp": "temperature",
    "temperature": "temperature",
    "pe": "poteng",
    "poteng": "poteng",
    "press": "press",
    "pressure": "press",
    "vol": "volume",
    "volume": "volume",
    "density": "density",
}
ORIGINAL_COLUMNS = [
    "step",
    "time",
    "temperature",
    "poteng",
    "press",
    "volume",
    "density",
]
OUTPUT_COLUMNS = [
    "direction",
    "source_model",
    "target_model",
    "N",
    "run",
    "step",
    "time",
    "temperature_source",
    "U_source",
    "U_target",
    "delta_U",
    "u_source",
    "u_target",
    "delta_u",
    "press_source",
    "volume_source",
    "density_source",
    "source_log",
    "rerun_log",
]
SUMMARY_COLUMNS = [
    "direction",
    "source_model",
    "target_model",
    "N",
    "run",
    "n_original_frames",
    "n_rerun_frames",
    "n_matched_frames",
    "mean_delta_U",
    "std_delta_U",
    "var_delta_U",
    "mean_delta_u",
    "std_delta_u",
    "var_delta_u",
    "mean_U_source_per_particle",
    "mean_U_target_per_particle",
]


@dataclass(frozen=True)
class Job:
    """Paths and metadata for one cross-energy alignment job."""

    source_model: str
    target_model: str
    size: int
    run: int
    original_log: Path
    rerun_log: Path

    @property
    def direction(self) -> str:
        return f"{self.source_model}_to_{self.target_model}"

    @property
    def label(self) -> str:
        return f"{self.direction}, N={self.size}, run={self.run}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing mW_NVT and NN_NVT (default: repository root)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Input/output analysis directory (default: PROJECT_ROOT/analysis)",
    )
    return parser.parse_args()


def discover_jobs(project_root: Path) -> list[Job]:
    """Discover source simulations and their expected rerun logs."""
    jobs: list[Job] = []
    for folder in sorted((project_root / "mW_NVT").glob("N_*")):
        match = SIZE_RE.fullmatch(folder.name)
        if folder.is_dir() and match:
            jobs.append(
                Job(
                    "mW",
                    "NNP",
                    int(match.group(1)),
                    1,
                    folder / "log.lammps",
                    folder / "log.rerun_NNP.lammps",
                )
            )

    for size_folder in sorted((project_root / "NN_NVT").glob("N_*")):
        size_match = SIZE_RE.fullmatch(size_folder.name)
        if not size_folder.is_dir() or not size_match:
            continue
        for run_folder in sorted(size_folder.glob("run_*")):
            run_match = RUN_RE.fullmatch(run_folder.name)
            if run_folder.is_dir() and run_match:
                jobs.append(
                    Job(
                        "NNP",
                        "mW",
                        int(size_match.group(1)),
                        int(run_match.group(1)),
                        run_folder / "log.lammps",
                        run_folder / "log.rerun_mW.lammps",
                    )
                )
    return sorted(jobs, key=lambda job: (job.direction, job.size, job.run))


def parse_thermo_log(path: Path) -> pd.DataFrame:
    """Parse numeric rows from every LAMMPS thermo table in a log."""
    records: list[dict[str, float]] = []
    header: list[str] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0].lower() == "step":
                lowered = [field.lower() for field in fields]
                if "pe" in lowered or "poteng" in lowered:
                    header = lowered
                else:
                    header = None
                continue

            if header is None or len(fields) != len(header):
                continue
            if not all(NUMBER_RE.fullmatch(value) for value in fields):
                continue

            record: dict[str, float] = {}
            for name, value in zip(header, fields):
                normalized = THERMO_COLUMNS.get(name)
                if normalized is not None:
                    record[normalized] = float(value)
            if "step" in record and "poteng" in record:
                records.append(record)

    frame = pd.DataFrame.from_records(records)
    for column in ORIGINAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[ORIGINAL_COLUMNS]


def load_retained_originals(path: Path) -> pd.DataFrame:
    """Load and normalize the previously cleaned original-thermo dataset."""
    frame = pd.read_csv(path)
    required = {"model", "N", "run", "step", "poteng"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    for column in ORIGINAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["model"] = frame["model"].astype(str)
    frame["N"] = pd.to_numeric(frame["N"], errors="raise").astype("int64")
    frame["run"] = pd.to_numeric(frame["run"], errors="raise").astype("int64")
    return frame


def original_for_job(job: Job, retained: pd.DataFrame | None) -> pd.DataFrame:
    """Select retained original rows, or parse the original log as fallback."""
    if retained is not None:
        selected = retained.loc[
            (retained["model"] == job.source_model)
            & (retained["N"] == job.size)
            & (retained["run"] == job.run),
            ORIGINAL_COLUMNS,
        ]
        return selected.copy()
    return parse_thermo_log(job.original_log)


def unique_unambiguous_steps(
    frame: pd.DataFrame, kind: str, job: Job
) -> pd.DataFrame:
    """Remove all occurrences of ambiguous duplicate Step keys."""
    if frame.empty:
        return frame.copy()
    duplicate_mask = frame.duplicated("step", keep=False)
    if duplicate_mask.any():
        duplicate_steps = sorted(frame.loc[duplicate_mask, "step"].unique())
        preview = ", ".join(f"{step:g}" for step in duplicate_steps[:10])
        suffix = " ..." if len(duplicate_steps) > 10 else ""
        warnings.warn(
            f"{job.label}: {kind} data contain {len(duplicate_steps)} duplicate "
            f"Step value(s) ({preview}{suffix}); excluding those ambiguous steps"
        )
    return frame.loc[~duplicate_mask].copy()


def step_preview(steps: set[float], limit: int = 10) -> str:
    ordered = sorted(steps)
    preview = ", ".join(f"{step:g}" for step in ordered[:limit])
    return preview + (" ..." if len(ordered) > limit else "")


def align_job(
    job: Job, original_raw: pd.DataFrame, rerun_raw: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate Step keys and inner-join one original/rerun pair."""
    n_original = len(original_raw)
    n_rerun = len(rerun_raw)
    original = unique_unambiguous_steps(original_raw, "original", job)
    rerun = unique_unambiguous_steps(rerun_raw, "rerun", job)

    original_steps = set(original["step"].tolist())
    rerun_steps = set(rerun["step"].tolist())
    missing_rerun = original_steps - rerun_steps
    extra_rerun = rerun_steps - original_steps
    if missing_rerun:
        warnings.warn(
            f"{job.label}: {len(missing_rerun)} original Step value(s) are missing "
            f"from rerun ({step_preview(missing_rerun)})"
        )
    if extra_rerun:
        warnings.warn(
            f"{job.label}: {len(extra_rerun)} rerun Step value(s) are absent from "
            f"the retained original data ({step_preview(extra_rerun)})"
        )

    merged = original.merge(
        rerun[["step", "poteng"]].rename(columns={"poteng": "U_target"}),
        on="step",
        how="inner",
        validate="one_to_one",
    ).sort_values("step", kind="stable")
    n_matched = len(merged)
    if n_matched == 0:
        warnings.warn(f"{job.label}: zero frames matched by Step")
    elif n_matched < len(original):
        warnings.warn(
            f"{job.label}: matched {n_matched} of {len(original)} retained, "
            "unambiguous original frames; fewer than expected"
        )

    result = merged.rename(
        columns={
            "temperature": "temperature_source",
            "poteng": "U_source",
            "press": "press_source",
            "volume": "volume_source",
            "density": "density_source",
        }
    ).copy()
    # Assigning metadata after constructing the frame avoids mixed scalar/Series
    # constructor behavior that differs between older pandas releases.
    result["direction"] = job.direction
    result["source_model"] = job.source_model
    result["target_model"] = job.target_model
    result["N"] = job.size
    result["run"] = job.run
    result["source_log"] = str(job.original_log.resolve())
    result["rerun_log"] = str(job.rerun_log.resolve())
    result["delta_U"] = result["U_target"] - result["U_source"]
    result["u_source"] = result["U_source"] / job.size
    result["u_target"] = result["U_target"] / job.size
    result["delta_u"] = result["delta_U"] / job.size
    result = result[OUTPUT_COLUMNS]

    # pandas uses sample standard deviation/variance (ddof=1), matching its
    # conventional group-summary behavior. Single-frame groups yield NaN.
    summary: dict[str, object] = {
        "direction": job.direction,
        "source_model": job.source_model,
        "target_model": job.target_model,
        "N": job.size,
        "run": job.run,
        "n_original_frames": n_original,
        "n_rerun_frames": n_rerun,
        "n_matched_frames": n_matched,
        "mean_delta_U": result["delta_U"].mean(),
        "std_delta_U": result["delta_U"].std(),
        "var_delta_U": result["delta_U"].var(),
        "mean_delta_u": result["delta_u"].mean(),
        "std_delta_u": result["delta_u"].std(),
        "var_delta_u": result["delta_u"].var(),
        "mean_U_source_per_particle": result["u_source"].mean(),
        "mean_U_target_per_particle": result["u_target"].mean(),
    }
    return result, summary


def save_outputs(
    cross_energy: pd.DataFrame, summary: pd.DataFrame, analysis_dir: Path
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "cross_energy_all.csv"
    summary_path = analysis_dir / "cross_energy_summary.csv"
    cross_energy.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(cross_energy):,} matched frames to {csv_path}")
    print(f"Saved {len(summary):,} job summaries to {summary_path}")

    if importlib.util.find_spec("pyarrow") is not None:
        parquet_path = analysis_dir / "cross_energy_all.parquet"
        try:
            cross_energy.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f"Saved {len(cross_energy):,} matched frames to {parquet_path}")
        except (ImportError, OSError, ValueError) as exc:
            # Older pandas versions can reject a newer pyarrow even though
            # pyarrow itself is fully able to write the table. Construct Arrow
            # arrays column-by-column to avoid pandas' compatibility adapter.
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pydict(
                    {
                        column: cross_energy[column].tolist()
                        for column in cross_energy.columns
                    }
                )
                pq.write_table(table, parquet_path)
                print(
                    f"Saved {len(cross_energy):,} matched frames to {parquet_path} "
                    "(direct pyarrow fallback)"
                )
            except (ImportError, OSError, TypeError, ValueError) as fallback_exc:
                warnings.warn(
                    f"Could not write Parquet output: pandas error: {exc}; "
                    f"direct pyarrow error: {fallback_exc}"
                )
    else:
        print("pyarrow is not installed; skipping Parquet output.")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    analysis_dir = (
        args.analysis_dir.resolve() if args.analysis_dir else project_root / "analysis"
    )
    retained_path = analysis_dir / "thermo_all.csv"
    try:
        retained = (
            load_retained_originals(retained_path) if retained_path.is_file() else None
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: cannot use retained thermo data: {exc}", file=sys.stderr)
        return 2

    if retained is not None:
        print(f"Using retained original trajectories from {retained_path}")
    else:
        print("No analysis/thermo_all.csv found; parsing original logs directly")

    jobs = discover_jobs(project_root)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    skipped = 0
    for job in jobs:
        if retained is None and not job.original_log.is_file():
            warnings.warn(f"{job.label}: missing original log {job.original_log}; skipping")
            skipped += 1
            continue
        if not job.rerun_log.is_file():
            warnings.warn(f"{job.label}: missing rerun log {job.rerun_log}; skipping")
            skipped += 1
            continue
        try:
            original = original_for_job(job, retained)
            rerun = parse_thermo_log(job.rerun_log)
        except (OSError, ValueError) as exc:
            warnings.warn(f"{job.label}: could not parse input: {exc}; skipping")
            skipped += 1
            continue
        if original.empty:
            warnings.warn(f"{job.label}: no retained original thermo rows")
        if rerun.empty:
            warnings.warn(f"{job.label}: no rerun thermo rows parsed")
        frame, summary = align_job(job, original, rerun)
        frames.append(frame)
        summaries.append(summary)
        print(
            f"{job.label}: original={len(original):,}, rerun={len(rerun):,}, "
            f"matched={len(frame):,}"
        )

    cross_energy = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    if not cross_energy.empty:
        cross_energy["step"] = pd.to_numeric(
            cross_energy["step"], errors="raise"
        ).astype("int64")
        cross_energy = cross_energy.sort_values(
            ["direction", "N", "run", "step"], kind="stable"
        ).reset_index(drop=True)
    summary_frame = pd.DataFrame.from_records(summaries, columns=SUMMARY_COLUMNS)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["direction", "N", "run"], kind="stable"
        ).reset_index(drop=True)

    save_outputs(cross_energy, summary_frame, analysis_dir)
    print(
        f"Processed {len(summaries)} of {len(jobs)} discovered jobs; "
        f"skipped {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
