#!/usr/bin/env python3
"""Prepare and analyze NNP reruns of every saved mW trajectory frame."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import collect_lammps_nvt as collector
import compute_energy_averages_and_reweighting as energy
from add_cross_poteng_to_thermo import parse_lammps_log
from prepare_cross_rerun import lammps_quote, make_initial_data, read_first_dump_frame


ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 128, 256, 512, 1024, 2048)


def dump_steps(path):
    """Read saved timestep labels without loading the coordinates into memory."""
    steps = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "ITEM: TIMESTEP":
                steps.append(int(next(handle).strip()))
    if len(steps) != len(set(steps)) or not steps:
        raise ValueError(f"Missing or duplicate dump steps: {path}")
    return steps


def prepare(output, nnp_dir):
    """Create isolated full-trajectory rerun inputs and a launcher."""
    for name in ("input.nn", "scaling.data", "weights.008.data"):
        if not (nnp_dir / name).is_file():
            raise FileNotFoundError(nnp_dir / name)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    launcher = ["#!/bin/bash", "set -euo pipefail"]
    for size in SIZES:
        trajectory = ROOT / "lammps/mW_NVT" / f"N_{size}" / "water.lammpstrj"
        steps = dump_steps(trajectory)
        frame = read_first_dump_frame(trajectory)
        if len(frame.atoms) != size or frame.timestep != steps[0]:
            raise ValueError(f"Invalid first frame: {trajectory}")
        job = output / f"N_{size}"
        job.mkdir()
        (job / "initial.data").write_text(make_initial_data(frame, trajectory))
        lines = [
            "# Evaluate every saved mW configuration with the NNP Hamiltonian",
            "echo both", "log log.rerun_NNP.lammps", "units metal",
            "atom_style atomic", "boundary p p p", "read_data initial.data",
            "mass 1 18.015",
            f"variable nnpDir string {lammps_quote(nnp_dir)}",
            "variable nnpCutoff equal 4.3065",
            'pair_style nnp dir ${nnpDir} showew no showewsum 1000000000000000 resetew yes maxew 1000000 calcd 20 emap "1:O"',
            "pair_coeff * * ${nnpCutoff}", "newton on", "neighbor 1.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "thermo_style custom step time temp pe etotal press vol density lx ly lz enthalpy",
            "thermo 1", "thermo_modify flush yes", "timestep 0.001",
            f"rerun {lammps_quote(trajectory)} dump x y z box yes", "",
        ]
        (job / "rerun.lmp").write_text("\n".join(lines))
        launcher.append(f'cd {lammps_quote(job)}')
        launcher.append('"$1" -in rerun.lmp -screen screen.out')
        print(f"Prepared N={size}: {len(steps)} frames, last step {steps[-1]}")
    (output / "run_reruns.sh").write_text("\n".join(launcher) + "\n")


def read_mw_size(size, output):
    """Join every saved mW frame to its source and new cross energy by step."""
    source_dir = ROOT / "lammps/mW_NVT" / f"N_{size}"
    steps = dump_steps(source_dir / "water.lammpstrj")
    source_log = source_dir / "log.lammps"
    if collector.classify_termination(source_log.read_text(errors="replace")) != "complete":
        raise ValueError(f"Source simulation did not finish cleanly: {source_log}")
    source = pd.DataFrame(collector.parse_thermo_blocks(source_log.read_text(errors="replace")))
    rerun_log = output / f"N_{size}" / "log.rerun_NNP.lammps"
    text = rerun_log.read_text(errors="replace")
    if "Total wall time:" not in text or re.search(r"^ERROR(?: on proc \d+)?:", text, re.M):
        raise ValueError(f"NNP rerun did not finish cleanly: {rerun_log}")
    atom_counts = [int(x) for x in re.findall(r"^\s*(\d+)\s+atoms\s*$", text, re.M)]
    if atom_counts != [size]:
        raise ValueError(f"Unexpected rerun atom count {atom_counts}: {rerun_log}")
    target = parse_lammps_log(rerun_log)
    for label, table in (("source", source), ("target", target)):
        table["step"] = table["step"].astype("int64")
        if table["step"].duplicated().any() or set(table["step"]) != set(steps):
            raise ValueError(f"{label} steps differ from the full dump at N={size}")
    merged = pd.DataFrame({"step": steps}).merge(source, on="step", validate="one_to_one")
    merged = merged.merge(target, on="step", validate="one_to_one")
    merged.insert(0, "model", "mW")
    merged.insert(1, "N", size)
    merged.insert(2, "run", 1)
    merged.insert(3, "source_run", 1)
    merged.insert(4, "sample_id", np.arange(len(merged)))
    merged["delta_U"] = merged["cross_poteng"] - merged["poteng"]
    merged["delta_u"] = merged["delta_U"] / size
    merged["poteng_per_particle"] = merged["poteng"] / size
    merged["cross_poteng_per_particle"] = merged["cross_poteng"] / size
    return merged


def analyze(output):
    """Build an isolated merged table, summary, and old-versus-full comparison."""
    old = pd.read_csv(
        ROOT / "results/full_mw_trajectory/thermo_all_with_cross_merged_first_2001.csv"
    )
    mw = pd.concat([read_mw_size(size, output) for size in SIZES], ignore_index=True)
    overlap = old.loc[old["model"] == "mW", ["N", "step", "poteng", "cross_poteng"]].merge(
        mw[["N", "step", "poteng", "cross_poteng"]], on=["N", "step"],
        suffixes=("_old", "_full"), validate="one_to_one",
    )
    if len(overlap) != len(old.loc[old["model"] == "mW"]):
        raise ValueError("Some previously analyzed mW frames are absent from the full reruns")
    verification = []
    for size, group in overlap.groupby("N"):
        source_difference = (group["poteng_old"] - group["poteng_full"]).abs().max()
        cross_difference = (group["cross_poteng_old"] - group["cross_poteng_full"]).abs().max()
        if source_difference > 1e-7 or cross_difference > 1e-5:
            raise ValueError(f"Old and full energies disagree at N={size}")
        verification.append({"N": size, "overlap_frames": len(group),
                             "max_source_difference_eV": source_difference,
                             "max_cross_difference_eV": cross_difference})
    pd.DataFrame(verification).to_csv(output / "overlap_verification.csv", index=False)
    # Reused source samples and NNP-to-mW cross energies stay identical.
    nnp = old.loc[old["model"] == "NNP"].copy()
    full = pd.concat([nnp, mw], ignore_index=True, sort=False)
    full.to_csv(output / "thermo_all_with_cross_merged.csv", index=False, float_format="%.10g")
    trimmed, removed = energy.remove_trajectory_endpoints(full)
    summary = energy.compute_summary(trimmed, 300.0)
    summary.to_csv(output / "energy_averages_and_reweighting.csv", index=False,
                   float_format="%.10g")
    removed.to_csv(output / "endpoint_removal.csv", index=False)
    prior = pd.read_csv(
        ROOT / "results/full_mw_trajectory/energy_averages_and_reweighting_first_2001.csv"
    )
    columns = ["N", "n_frames_mW", "U_mW_in_mW", "U_NNP_in_mW",
               "U_NNP_reweighted_from_mW",
               "error_before_mW_to_NNP", "error_after_mW_to_NNP",
               "abs_error_before_mW_to_NNP", "abs_error_after_mW_to_NNP",
               "abs_error_before_NNP_to_mW", "abs_error_after_NNP_to_mW",
               "neff_mW_to_NNP", "neff_fraction_mW_to_NNP",
               "max_weight_fraction_mW_to_NNP"]
    comparison = prior[columns].merge(summary[columns], on="N", suffixes=("_old", "_full"))
    comparison.to_csv(output / "comparison.csv", index=False, float_format="%.10g")
    print(comparison.to_string(index=False))
    print(f"Saved full analysis to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/full_mw_trajectory")
    parser.add_argument("--nnp-dir", type=Path, default=ROOT.parent / "nnp-water")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.output_dir.resolve(), args.nnp_dir.resolve())
        else:
            analyze(args.output_dir.resolve())
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
