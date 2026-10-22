#!/usr/bin/env python3
"""Extract mW thermo rows matching each retained 2001-frame trajectory."""

import argparse
from pathlib import Path


def trajectory_steps(path):
    """Read timestep labels without loading the large trajectory into memory."""
    steps = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "ITEM: TIMESTEP":
                steps.append(int(next(handle).strip()))
    if len(steps) != 2001 or len(set(steps)) != len(steps):
        raise ValueError(f"Expected 2001 distinct trajectory steps: {path}")
    return steps


def thermo_rows(path):
    """Read the Step thermo table from a LAMMPS log."""
    header = None
    rows = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] == "Step" and "PotEng" in fields:
                if header is not None:
                    raise ValueError(f"Multiple thermo tables in {path}")
                header = line
                continue
            if header is None:
                continue
            if not fields or not fields[0].isdigit():
                if rows:
                    break
                continue
            step = int(fields[0])
            if step in rows:
                raise ValueError(f"Duplicate thermo step {step} in {path}")
            rows[step] = line
    if header is None:
        raise ValueError(f"No Step/PotEng table in {path}")
    return header, rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "lammps/mW_NVT")
    parser.add_argument("--output-root", type=Path, default=root / "results/mw_nvt_first_2001")
    args = parser.parse_args()
    trajectories = sorted(args.input_root.glob("N_*/water_first_2001.lammpstrj"))
    if not trajectories:
        parser.error(f"No retained trajectories found under {args.input_root}")
    for trajectory in trajectories:
        steps = trajectory_steps(trajectory)
        header, rows = thermo_rows(trajectory.with_name("log.lammps"))
        missing = set(steps) - rows.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} thermo steps for {trajectory}")
        output = args.output_root / trajectory.parent.name / "log_first_2001.lammps"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the original LAMMPS header and row formatting for downstream parsers.
        output.write_text(header + "".join(rows[step] for step in steps))
        print(f"{output}: {len(steps)} rows, steps {steps[0]} to {steps[-1]}")


if __name__ == "__main__":
    main()
