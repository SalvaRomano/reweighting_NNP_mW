#!/usr/bin/env python3
"""Prepare corrected mW single-point reruns, or analyze their NNP-only output."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from add_cross_poteng_to_thermo import parse_lammps_log, normalize_integer_column
from compute_energy_averages_and_reweighting import (
    KB_EV_PER_K, load_and_validate, remove_trajectory_endpoints, reweighted_average,
)
from prepare_cross_rerun import read_first_dump_frame, make_initial_data, lammps_quote


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "analyze"])
    parser.add_argument("--source", type=Path,
                        default=root / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--trajectory-root", type=Path, default=root / "lammps/NN_NVT")
    parser.add_argument("--potential", type=Path, default=root / "lammps/mW_NVT/mW.sw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/corrected_mw_rerun")
    parser.add_argument("--lammps", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = load_and_validate(args.source.resolve())
    source = source.loc[source.model == "NNP"].copy()
    source["step"] = source.step.astype("int64")
    if source.empty or source.duplicated(["N", "source_run", "step"]).any():
        raise ValueError("Source must contain uniquely keyed NNP frames")

    if args.mode == "prepare":
        if args.lammps is None or not args.lammps.is_file():
            parser.error("prepare requires --lammps pointing to an executable")
        # Preserve the precise Hamiltonian used even if the source file changes later.
        output.mkdir(parents=True, exist_ok=False)
        potential = args.potential.resolve().read_bytes()
        (output / "mW.sw").write_bytes(potential)
        manifest = {"source": str(args.source.resolve()),
                    "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    "potential_source": str(args.potential.resolve()),
                    "potential_sha256": hashlib.sha256(potential).hexdigest(),
                    "temperature_K": 300, "jobs": []}
        commands = ["#!/bin/bash", "set -euo pipefail", "export OMP_NUM_THREADS=1"]
        for (size, run), group in source.groupby(["N", "source_run"], sort=True):
            trajectory = (args.trajectory_root / f"N_{size}/run_{run}/water.lammpstrj").resolve()
            first = read_first_dump_frame(trajectory)
            if len(first.atoms) != size:
                raise ValueError(f"Atom count mismatch: {trajectory}")
            folder = output / f"N_{size}/run_{run}"
            folder.mkdir(parents=True)
            (folder / "initial.data").write_text(make_initial_data(first, trajectory))
            # Stop at the final retained source step, avoiding later failed MD frames.
            text = "\n".join([
                "units metal", "atom_style atomic", "boundary p p p",
                f"read_data {lammps_quote(folder / 'initial.data')}",
                "mass 1 18.015", "pair_style sw",
                f"pair_coeff * * {lammps_quote(output / 'mW.sw')} mW",
                "neighbor 1.0 bin", "neigh_modify delay 0 every 1 check yes",
                "thermo_style custom step pe", "thermo 1",
                "thermo_modify format float %.16g flush yes",
                f"rerun {lammps_quote(trajectory)} last {int(group.step.max())} dump x y z box yes",
                "",
            ])
            (folder / "rerun.lmp").write_text(text)
            command = [str(args.lammps.resolve()), "-in", str(folder / "rerun.lmp"),
                       "-log", str(folder / "log.rerun_mW.lammps"),
                       "-screen", str(folder / "screen.out")]
            commands.extend([f"echo 'N={size} run={run}'", shlex.join(command)])
            manifest["jobs"].append({"N": int(size), "source_run": int(run),
                                     "trajectory": str(trajectory),
                                     "last_step": int(group.step.max()),
                                     "n_source_frames": len(group)})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "run_reruns.sh").write_text("\n".join(commands) + "\n")
        print(f"Prepared {len(manifest['jobs'])} single-point reruns: {output / 'run_reruns.sh'}")
        return

    manifest = json.loads((output / "manifest.json").read_text())
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Source table changed since preparation")
    if hashlib.sha256((output / "mW.sw").read_bytes()).hexdigest() != manifest["potential_sha256"]:
        raise ValueError("Potential snapshot changed since preparation")
    frames = []
    for (size, run), group in source.groupby(["N", "source_run"], sort=True):
        log = output / f"N_{size}/run_{run}/log.rerun_mW.lammps"
        text = log.read_text()
        if "Total wall time:" not in text or "ERROR" in text:
            raise ValueError(f"Rerun did not finish cleanly: {log}")
        target = parse_lammps_log(log)
        target["step"] = normalize_integer_column(target.step, "rerun step")
        group = group.drop(columns=["cross_poteng", "delta_U", "delta_u",
                                    "poteng_per_particle", "cross_poteng_per_particle"],
                           errors="ignore")
        paired = group.merge(target, on="step", how="left", validate="one_to_one")
        if not np.isfinite(paired.cross_poteng).all():
            raise ValueError(f"Missing or invalid target energies: {log}")
        paired["delta_U"] = paired.cross_poteng - paired.poteng
        paired["delta_u"] = paired.delta_U / paired.N
        paired["poteng_per_particle"] = paired.poteng / paired.N
        paired["cross_poteng_per_particle"] = paired.cross_poteng / paired.N
        frames.append(paired)
    corrected = pd.concat(frames, ignore_index=True)
    corrected.to_csv(output / "nnp_frames_with_corrected_mw.csv", index=False, float_format="%.16g")
    trimmed, removal = remove_trajectory_endpoints(corrected)
    removal.to_csv(output / "endpoint_removal.csv", index=False)
    # Retain the previous frame selection so changes reflect the new Hamiltonian.
    beta = 1 / (KB_EV_PER_K * 300)
    for groups, filename in [(["N"], "summary_by_size.csv"),
                              (["N", "source_run"], "summary_by_run.csv")]:
        records = []
        for key, group in trimmed.groupby(groups, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            result = reweighted_average(group, beta)
            records.append({**dict(zip(groups, key)), "n_frames": len(group),
                            "temperature_K": 300,
                            "mean_source_energy_eV_per_particle": (group.poteng / group.N).mean(),
                            "mean_delta_U_eV": group.delta_U.mean(),
                            "std_delta_U_eV": group.delta_U.std(ddof=0),
                            "reweighted_mw_energy_eV_per_particle": result.pop("energy"),
                            **result})
        summary = pd.DataFrame(records)
        summary.to_csv(output / filename, index=False, float_format="%.16g")
        if len(groups) == 1:
            print(summary[["N", "n_frames", "neff", "neff_fraction"]].to_string(index=False))
    print(f"Matched {len(corrected)} NNP frames; analyzed {len(trimmed)} after endpoint removal.")


if __name__ == "__main__":
    main()
