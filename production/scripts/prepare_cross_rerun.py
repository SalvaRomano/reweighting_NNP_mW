#!/usr/bin/env python3
"""Prepare LAMMPS jobs that evaluate trajectories with the other Hamiltonian.

The target-model LAMMPS inputs supplied by the user are treated as templates.
Their force-field and general settings are retained, while dynamics and output
commands are disabled and replaced by a ``rerun`` command.  A minimal LAMMPS
data file is built from the first trajectory frame because ``rerun`` requires
the simulation box and atom IDs to exist before it reads dump snapshots.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


SIZE_RE = re.compile(r"N_(\d+)$")
RUN_RE = re.compile(r"run_(\d+)$")
DISABLED_COMMANDS = {
    "dump",
    "dump_modify",
    "fix",
    "minimize",
    "restart",
    "run",
    "timestep",
    "unfix",
    "velocity",
    "write_data",
    "write_dump",
    "write_restart",
}
INDEX_FIELDS = [
    "direction",
    "source_model",
    "target_model",
    "system_size",
    "run",
    "original_trajectory",
    "generated_md_lmp",
    "output_folder",
]


@dataclass(frozen=True)
class Simulation:
    """One discovered source simulation."""

    model: str
    size: int
    run: int
    folder: Path


@dataclass(frozen=True)
class DumpFrame:
    """The box and atomic coordinates from one LAMMPS dump snapshot."""

    timestep: int
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    tilt: tuple[float, float, float] | None
    atoms: tuple[tuple[int, int, float, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mw-template",
        type=Path,
        required=True,
        help="Working mW md.lmp template (used for NNP_to_mW jobs)",
    )
    parser.add_argument(
        "--nnp-template",
        type=Path,
        required=True,
        help="Working NNP md.lmp template (used for mW_to_NNP jobs)",
    )
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
        help="Destination directory (default: PROJECT_ROOT/cross_energy)",
    )
    parser.add_argument(
        "--trajectory-name",
        default="water.trj",
        help="Preferred trajectory filename (default: water.trj)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated md.lmp, initial.data, and jobs.csv files",
    )
    return parser.parse_args()


def discover_simulations(project_root: Path) -> list[Simulation]:
    """Discover valid mW and NNP simulation directories."""
    simulations: list[Simulation] = []

    for folder in sorted((project_root / "mW_NVT").glob("N_*")):
        match = SIZE_RE.fullmatch(folder.name)
        if folder.is_dir() and match:
            simulations.append(Simulation("mW", int(match.group(1)), 1, folder))

    for size_folder in sorted((project_root / "NN_NVT").glob("N_*")):
        size_match = SIZE_RE.fullmatch(size_folder.name)
        if not size_folder.is_dir() or not size_match:
            continue
        for run_folder in sorted(size_folder.glob("run_*")):
            run_match = RUN_RE.fullmatch(run_folder.name)
            if run_folder.is_dir() and run_match:
                simulations.append(
                    Simulation(
                        "NNP",
                        int(size_match.group(1)),
                        int(run_match.group(1)),
                        run_folder,
                    )
                )

    return sorted(simulations, key=lambda item: (item.model, item.size, item.run))


def find_trajectory(folder: Path, preferred_name: str) -> tuple[Path | None, str | None]:
    """Find the requested trajectory, accepting the repository's legacy name."""
    preferred = folder / preferred_name
    if preferred.is_file():
        return preferred.resolve(), None

    if preferred_name == "water.trj":
        fallback = folder / "water.lammpstrj"
        if fallback.is_file():
            return fallback.resolve(), f"using {fallback.name} instead of water.trj"
    return None, None


def read_first_dump_frame(path: Path) -> DumpFrame:
    """Read the first frame of a LAMMPS custom dump without loading the file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if handle.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError("first line is not 'ITEM: TIMESTEP'")
        timestep = int(handle.readline().strip())
        if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("missing NUMBER OF ATOMS section")
        atom_count = int(handle.readline().strip())

        box_header = handle.readline().split()
        if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
            raise ValueError("missing BOX BOUNDS section")
        triclinic = all(name in box_header[3:] for name in ("xy", "xz", "yz"))
        raw_bounds = [tuple(map(float, handle.readline().split())) for _ in range(3)]

        atom_header = handle.readline().split()
        if atom_header[:2] != ["ITEM:", "ATOMS"]:
            raise ValueError("missing ATOMS section")
        columns = atom_header[2:]
        required = ("id", "type", "x", "y", "z")
        if not all(column in columns for column in required):
            raise ValueError("ATOMS columns must include id type x y z")
        indices = [columns.index(column) for column in required]

        atoms: list[tuple[int, int, float, float, float]] = []
        for _ in range(atom_count):
            fields = handle.readline().split()
            if not fields:
                raise ValueError("trajectory ended inside the first frame")
            atoms.append(
                (
                    int(fields[indices[0]]),
                    int(fields[indices[1]]),
                    float(fields[indices[2]]),
                    float(fields[indices[3]]),
                    float(fields[indices[4]]),
                )
            )

    if triclinic:
        xlo_bound, xhi_bound, xy = raw_bounds[0]
        ylo_bound, yhi_bound, xz = raw_bounds[1]
        zlo, zhi, yz = raw_bounds[2]
        # LAMMPS dump files store triclinic bounding-box extents, whereas data
        # files require the true restricted-triclinic x/y limits.
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        bounds = ((xlo, xhi), (ylo, yhi), (zlo, zhi))
        tilt = (xy, xz, yz)
    else:
        if any(len(bound) != 2 for bound in raw_bounds):
            raise ValueError("unsupported BOX BOUNDS format")
        bounds = tuple((bound[0], bound[1]) for bound in raw_bounds)
        tilt = None

    atoms.sort(key=lambda atom: atom[0])
    return DumpFrame(timestep, bounds, tilt, tuple(atoms))


def make_initial_data(frame: DumpFrame, source: Path) -> str:
    """Build the minimal atomic-style data file needed to initialize rerun."""
    atom_types = max(atom[1] for atom in frame.atoms)
    lines = [
        f"LAMMPS data file initialized from first frame of {source}",
        "",
        f"{len(frame.atoms)} atoms",
        f"{atom_types} atom types",
        "",
    ]
    for axis, (lower, upper) in zip("xyz", frame.bounds):
        lines.append(f"{lower:.16g} {upper:.16g} {axis}lo {axis}hi")
    if frame.tilt is not None:
        lines.append("{:.16g} {:.16g} {:.16g} xy xz yz".format(*frame.tilt))
    lines.extend(["", "Atoms # atomic", ""])
    lines.extend(
        f"{atom_id} {atom_type} {x:.16g} {y:.16g} {z:.16g}"
        for atom_id, atom_type, x, y, z in frame.atoms
    )
    return "\n".join(lines) + "\n"


def lammps_quote(path: Path) -> str:
    """Quote a path for LAMMPS input syntax."""
    value = str(path.resolve())
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def transform_template(template: str, initial_data: Path, trajectory: Path) -> str:
    """Conservatively turn a dynamics template into a rerun input."""
    output: list[str] = []
    found_read = False
    found_thermo_style = False
    found_thermo = False

    for line in template.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        try:
            command = shlex.split(stripped, comments=True)[0].lower()
        except (ValueError, IndexError):
            output.append(line)
            continue

        indent = line[: len(line) - len(stripped)]
        if command == "read_data":
            output.append(f"{indent}read_data {lammps_quote(initial_data)}")
            found_read = True
        elif command == "replicate":
            output.append(f"{indent}# CROSS_RERUN disabled: {stripped}")
        elif command == "thermo_style":
            output.append(f"{indent}thermo_style custom step pe")
            found_thermo_style = True
        elif command == "thermo":
            output.append(f"{indent}thermo 1")
            found_thermo = True
        elif command in DISABLED_COMMANDS:
            output.append(f"{indent}# CROSS_RERUN disabled: {stripped}")
        else:
            output.append(line)

    if not found_read:
        raise ValueError("template has no active read_data command to replace")
    output.extend(["", "# Cross-Hamiltonian single-point evaluations"])
    if not found_thermo_style:
        output.append("thermo_style custom step pe")
    if not found_thermo:
        output.append("thermo 1")
    output.append(f"rerun {lammps_quote(trajectory)} dump x y z box yes")
    return "\n".join(output) + "\n"


def write_text_safely(path: Path, text: str, force: bool) -> bool:
    """Write text unless protected by the no-overwrite policy."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def write_index(path: Path, rows: list[dict[str, object]], force: bool) -> bool:
    """Write the job index using stable columns and ordering."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_dir.resolve() if args.output_dir else project_root / "cross_energy"
    )
    templates = {"mW": args.mw_template.resolve(), "NNP": args.nnp_template.resolve()}
    for model, path in templates.items():
        if not path.is_file():
            print(f"ERROR: {model} template does not exist: {path}", file=sys.stderr)
            return 2

    template_text = {
        model: path.read_text(encoding="utf-8", errors="replace")
        for model, path in templates.items()
    }
    simulations = discover_simulations(project_root)
    if not simulations:
        print(f"WARNING: no simulations found below {project_root}", file=sys.stderr)

    rows: list[dict[str, object]] = []
    created = skipped_existing = missing = invalid = fallbacks = 0
    for simulation in simulations:
        trajectory, note = find_trajectory(simulation.folder, args.trajectory_name)
        label = f"{simulation.model} N={simulation.size} run={simulation.run}"
        if trajectory is None:
            print(
                f"WARNING: skipping {label}; missing "
                f"{simulation.folder / args.trajectory_name}",
                file=sys.stderr,
            )
            missing += 1
            continue
        if note:
            print(f"WARNING: {label}: {note}", file=sys.stderr)
            fallbacks += 1

        target_model = "mW" if simulation.model == "NNP" else "NNP"
        direction = f"{simulation.model}_to_{target_model}"
        job_folder = output_root / direction / f"N_{simulation.size}" / f"run_{simulation.run}"
        data_path = job_folder / "initial.data"
        input_path = job_folder / "md.lmp"
        try:
            frame = read_first_dump_frame(trajectory)
            if len(frame.atoms) != simulation.size:
                raise ValueError(
                    f"first frame contains {len(frame.atoms)} atoms, expected {simulation.size}"
                )
            data_text = make_initial_data(frame, trajectory)
            input_text = transform_template(
                template_text[target_model], data_path, trajectory
            )
        except (OSError, ValueError) as exc:
            print(f"WARNING: skipping {label}; {exc}", file=sys.stderr)
            invalid += 1
            continue

        data_written = write_text_safely(data_path, data_text, args.force)
        input_written = write_text_safely(input_path, input_text, args.force)
        if data_written and input_written:
            created += 1
            status = "created"
        else:
            skipped_existing += 1
            status = "kept existing file(s)"
        print(f"{status:21} {direction}/N_{simulation.size}/run_{simulation.run}")

        rows.append(
            {
                "direction": direction,
                "source_model": simulation.model,
                "target_model": target_model,
                "system_size": simulation.size,
                "run": simulation.run,
                "original_trajectory": str(trajectory),
                "generated_md_lmp": str(input_path.resolve()),
                "output_folder": str(job_folder.resolve()),
            }
        )

    index_path = output_root / "jobs.csv"
    index_written = write_index(index_path, rows, args.force)
    index_status = "written" if index_written else "kept (use --force to replace)"
    print("\nSummary")
    print(f"  simulations discovered: {len(simulations)}")
    print(f"  jobs indexed:           {len(rows)}")
    print(f"  jobs created:           {created}")
    print(f"  existing jobs kept:     {skipped_existing}")
    print(f"  missing trajectories:   {missing}")
    print(f"  invalid trajectories:   {invalid}")
    print(f"  legacy-name fallbacks:  {fallbacks}")
    print(f"  index: {index_path} ({index_status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
