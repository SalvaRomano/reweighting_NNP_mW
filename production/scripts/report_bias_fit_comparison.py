#!/usr/bin/env python3
"""Compare zero-intercept inverse-n bias fits using fixed full-source theory."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ratio_bias import weighted_inverse_n_fit  # noqa: E402


FIT_CUTOFFS = (100, 200)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 512))
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/bias_fit_comparison.csv")
    return parser.parse_args()


def build_report(sizes, analysis_dir):
    rows = []
    for size in sizes:
        bootstrap = pd.read_csv(analysis_dir / f"bias_N{size}.csv")
        summary = pd.read_csv(analysis_dir / f"bias_N{size}_summary.csv")
        if summary["direction"].duplicated().any():
            raise ValueError(f"N={size}: duplicate direction in full-source summary")
        theory_by_direction = summary.set_index("direction")["B_theory"]

        for direction, direction_data in bootstrap.groupby("direction", sort=False):
            if direction not in theory_by_direction:
                raise ValueError(f"N={size}: no full-source B_theory for {direction}")
            # Read B_theory once from the full-source moments. It is copied into
            # both fit rows; it is never recomputed from a fitted n subset.
            theory = float(theory_by_direction.loc[direction])
            for cutoff in FIT_CUTOFFS:
                selected = direction_data.loc[direction_data["n"] >= cutoff]
                coefficient, coefficient_sem = weighted_inverse_n_fit(
                    selected["n"], selected["bias_empirical"],
                    selected["sem_theta_hat"])
                rows.append({
                    "N": size,
                    "direction": direction,
                    "B_theory": theory,
                    "n_min": cutoff,
                    "B_empirical": coefficient,
                    "B_empirical_sem": coefficient_sem,
                    "B_empirical_over_B_theory": coefficient / theory,
                })

    report = pd.DataFrame(rows)
    theory_counts = report.groupby(["N", "direction"])["B_theory"].nunique(dropna=False)
    if not (theory_counts == 1).all():
        failures = theory_counts.loc[theory_counts != 1]
        raise RuntimeError(
            "IMPLEMENTATION ERROR: B_theory changes with n_min for "
            + ", ".join(f"N={n} {direction}" for n, direction in failures.index)
        )
    return report


def main():
    args = parse_args()
    report = build_report(args.sizes, args.analysis_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, float_format="%.15g")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    print("\nB_theory consistency check: PASS (identical across n_min per N/direction)")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
