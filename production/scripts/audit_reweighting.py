#!/usr/bin/env python3
"""Reproduce the existing analysis and write independent audit diagnostics."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import add_cross_poteng_to_thermo as cross
import collect_lammps_nvt as collect
import compute_energy_averages_and_reweighting as energy
import merge_nnp_runs as pooling
from prepare_cross_rerun import read_first_dump_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--comparison-log", type=Path, default=None,
                        help="Two successive N=64/run_1 SW reruns: production, then dataset")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "results" / "reweighting_audit"
    output.mkdir(parents=True, exist_ok=True)

    # Rebuild every join from raw logs without calling production output writers.
    with warnings.catch_warnings(record=True) as caught:
        master = collect.collect(root / "lammps")
        reruns = cross.parse_all_rerun_logs(
            root / "results" / "full_mw_trajectory",
            root / "results" / "corrected_mw_rerun", master)
        rebuilt = cross.add_cross_poteng(master, reruns)
        cross.validate_matches(master, reruns, rebuilt, False)
    (output / "collection_warnings.txt").write_text(
        "\n".join(str(w.message) for w in caught) + "\n")
    stored = pd.read_csv(root / "analysis" / "thermo_all_with_cross.csv")
    joined = rebuilt.merge(stored, on=cross.KEY_COLUMNS, how="outer",
                           suffixes=("_raw", "_stored"), validate="one_to_one",
                           indicator=True)
    assert joined["_merge"].eq("both").all(), "Stored and raw frame keys differ"
    checks = {"matched_rows": len(joined)}
    for column in ["poteng", "cross_poteng"]:
        difference = (joined[column + "_raw"] - joined[column + "_stored"]).abs()
        checks["max_abs_difference_" + column] = float(difference.max())
        # The stored CSV keeps 10 significant digits; at |U| ~ 900 eV that is
        # a rounding error up to 5e-8 eV, so allow 1e-6 eV (~4e-5 kT).
        assert difference.max() < 1e-6, "Stored and raw energies differ"
    merged = pooling.merge_nnp_runs(rebuilt)
    pooling.validate_merged(rebuilt, merged)
    trimmed, removal = energy.remove_trajectory_endpoints(merged)
    removal.to_csv(output / "endpoint_removal.csv", index=False)
    summary = energy.compute_summary(trimmed, 300.0)
    previous = pd.read_csv(root / "analysis" / "energy_averages_and_reweighting.csv")
    for column in ["neff_NNP_to_mW", "neff_mW_to_NNP"]:
        reference = previous.set_index("N")[column]
        difference = (summary.set_index("N")[column] - reference).abs()
        checks["max_abs_difference_" + column] = float(difference.max())
        # Relative check: 10-digit energy rounding in the stored tables shifts
        # log-weights by ~1e-6 and N_eff by up to ~2e-7 relative.
        relative = difference / reference.abs()
        assert relative.notna().all() and relative.max() < 1e-6
    summary.to_csv(output / "recomputed_summary.csv", index=False)

    diagnostics = []
    for (model, size), group in trimmed.groupby(["model", "N"]):
        result = energy.reweighted_average(group, 1 / (energy.KB_EV_PER_K * 300))
        diagnostics.append({"model": model, "N": size, "frames": len(group),
                            "std_delta_U_eV": group.delta_U.std(ddof=0),
                            "std_delta_U_over_kBT": group.delta_U.std(ddof=0)
                            / (energy.KB_EV_PER_K * 300), **result})
    pd.DataFrame(diagnostics).to_csv(output / "weight_diagnostics.csv", index=False)

    trajectories = []
    for model, size, run, path in collect.discover_runs(root / "lammps"):
        first = read_first_dump_frame(path.parent / "water.lammpstrj")
        trajectories.append({"model": model, "N": size, "run": run,
                             "termination": collect.classify_termination(path.read_text()),
                             "first_frame_atoms": len(first.atoms),
                             **{f"L{axis}": hi - lo for axis, (lo, hi)
                                in zip("xyz", first.bounds)}})
    pd.DataFrame(trajectories).to_csv(output / "trajectory_checks.csv", index=False)

    if args.comparison_log:
        # A decreasing step separates the two evaluations of the same trajectory.
        evaluated = cross.parse_lammps_log(args.comparison_log)
        evaluated["step"] = cross.normalize_integer_column(evaluated.step, "rerun step")
        boundaries = np.flatnonzero(np.diff(evaluated.step) < 0) + 1
        assert len(boundaries) == 1, "Expected exactly two successive reruns"
        split = int(boundaries[0])
        selected = trimmed.loc[(trimmed.model == "NNP") & (trimmed.N == 64)
                               & (trimmed.source_run == 1)]
        comparisons = []
        for label, target in zip(["production_SW", "dataset_SW"],
                                 [evaluated.iloc[:split], evaluated.iloc[split:]]):
            paired = selected.drop(columns="cross_poteng").merge(
                target, on="step", how="left", validate="one_to_one")
            assert np.isfinite(paired.cross_poteng).all()
            paired["delta_U"] = paired.cross_poteng - paired.poteng
            result = energy.reweighted_average(paired, 1 / (energy.KB_EV_PER_K * 300))
            comparisons.append({"target": label, "frames": len(paired),
                                "mean_delta_U_eV": paired.delta_U.mean(),
                                "std_delta_U_eV": paired.delta_U.std(ddof=0), **result})
        comparison = pd.DataFrame(comparisons)
        comparison.to_csv(output / "sw_comparison.csv", index=False)
        print(comparison[["target", "frames", "std_delta_U_eV", "neff", "neff_fraction"]]
              .to_string(index=False))
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print(f"Audit outputs: {output}")


if __name__ == "__main__":
    main()
