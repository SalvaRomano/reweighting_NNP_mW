#!/usr/bin/env python3
"""Prepare NNP single-point reruns of the retained corrected-mW trajectories."""

import argparse
from pathlib import Path

from prepare_cross_rerun import lammps_quote, make_initial_data, read_first_dump_frame
from truncate_mw_logs import trajectory_steps


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=root / "lammps/mW_NVT")
    parser.add_argument("--nnp-dir", type=Path, default=root.parent / "nnp-water")
    args = parser.parse_args()
    nnp_dir = args.nnp_dir.resolve()
    for name in ("input.nn", "scaling.data", "weights.008.data"):
        if not (nnp_dir / name).is_file():
            parser.error(f"Missing NNP file: {nnp_dir / name}")

    trajectories = sorted(args.trajectory_root.glob("N_*/water_first_2001.lammpstrj"))
    if not trajectories:
        parser.error(f"No retained mW trajectories under {args.trajectory_root}")
    for trajectory in trajectories:
        size = int(trajectory.parent.name[2:])
        steps = trajectory_steps(trajectory)
        frame = read_first_dump_frame(trajectory)
        if len(frame.atoms) != size or frame.timestep != steps[0]:
            raise ValueError(f"Invalid first frame: {trajectory}")
        output = trajectory.parent / "rerun_NNP"
        data = output / "initial.data"
        md = output / "md.lmp"
        if data.exists() or md.exists():
            raise FileExistsError(f"Rerun input already exists in {output}")
        lines = [
            "# Evaluate retained corrected-mW configurations with the NNP Hamiltonian",
            "echo both",
            "log log.rerun_NNP.lammps",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "read_data initial.data",
            "mass 1 18.015",
            f"variable nnpDir string {lammps_quote(nnp_dir)}",
            "variable nnpCutoff equal 4.3065",
            'pair_style nnp dir ${nnpDir} showew no showewsum 1000000000000000 resetew yes maxew 1000000 calcd 20 emap "1:O"',
            "pair_coeff * * ${nnpCutoff}",
            "newton on",
            "neighbor 1.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "thermo_style custom step time temp pe etotal press vol density lx ly lz enthalpy",
            "thermo 1",
            "thermo_modify flush yes",
            "timestep 0.001",
            f"rerun {lammps_quote(trajectory)} dump x y z box yes",
            "",
        ]
        output.mkdir(parents=True, exist_ok=True)
        data.write_text(make_initial_data(frame, trajectory))
        md.write_text("\n".join(lines))
        print(f"{md}: {size} atoms, {len(steps)} frames, last step {steps[-1]}")


if __name__ == "__main__":
    main()
