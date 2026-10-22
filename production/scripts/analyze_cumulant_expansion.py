#!/usr/bin/env python3
"""Compute exponential-reweighting cumulants by N and direction.

Writes the fourth-order summary (with bootstrap standard errors of C1-C4)
and the empirical Taylor series of the sample tilted mean through a higher
order (default 12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cumulant_expansion import (contributions, prepare_observables,  # noqa: E402
                                    taylor_coefficients, tilted_mean)
import compute_energy_averages_and_reweighting as energy  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulant_expansion/summary.csv")
    parser.add_argument("--series-output", type=Path,
                        default=ROOT / "results/cumulant_expansion/taylor_series_order12.csv")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Nominal NVT temperature in K for the single beta (default: 300)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Frame resamples for the C1-C4 standard errors (default: 1000)")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-order", type=int, default=12,
                        help="Highest order of the empirical Taylor series (default: 12)")
    return parser.parse_args()


def bootstrap_standard_errors(V, G, repetitions, rng):
    """Standard errors of C1-C4 from resampling the source frames.

    This is the statistical uncertainty of each coefficient as an estimate of
    the population cumulant, given the finite set of (independent) frames.
    """
    estimates = np.empty((repetitions, 4))
    for repetition in range(repetitions):
        index = rng.integers(0, len(V), len(V))
        terms = contributions(V[index], G[index])
        estimates[repetition] = [terms[f"C{order}"] for order in range(1, 5)]
    return estimates.std(axis=0, ddof=1)


def analyze(frame, temperature, repetitions, rng, max_order):
    """Use the same endpoint convention as the existing reweighting summary."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Nominal temperature must be finite and positive")
    beta = 1.0 / (energy.KB_EV_PER_K * temperature)
    trimmed, removal = energy.remove_trajectory_endpoints(frame)
    records = []
    series_records = []
    for size in sorted(trimmed["N"].unique()):
        by_model = {
            model: trimmed.loc[(trimmed["N"] == size) & (trimmed["model"] == model)]
            for model in ("NNP", "mW")
        }
        if any(group.empty for group in by_model.values()):
            raise ValueError(f"Both source ensembles are required at N={size}")
        for source, target in (("NNP", "mW"), ("mW", "NNP")):
            group = by_model[source]
            target_group = by_model[target]
            V, G = prepare_observables(
                group["poteng"].to_numpy(dtype=float),
                group["cross_poteng"].to_numpy(dtype=float),
                int(size), beta,
            )
            terms = contributions(V, G)
            full = tilted_mean(V, G)
            direct = float(np.mean(target_group["poteng"].to_numpy(dtype=float) / size))
            record = {
                "N": int(size), "direction": f"{source}_to_{target}",
                "source_model": source, "target_model": target,
                "temperature_K": temperature, "n_source_frames": len(group),
                "n_direct_target_frames": len(target_group),
                **terms,
                "G_full_reweighted": full, "G_direct_target": direct,
            }
            for order in range(5):
                value = terms[f"G_order{order}"]
                record[f"error_vs_full_reweighting_order{order}"] = value - full
                record[f"error_vs_direct_target_order{order}"] = value - direct
            standard_errors = bootstrap_standard_errors(V, G, repetitions, rng)
            for order, error in zip(range(1, 5), standard_errors):
                record[f"C{order}_se"] = error
                record[f"C{order}_abs_over_se"] = abs(terms[f"C{order}"]) / error
            record["n_bootstrap"] = repetitions
            records.append(record)

            # Empirical Taylor series of this finite sample. Its partial sums
            # converge (if at all) to this same sample's full exponentially
            # reweighted mean; they say nothing about population uncertainty.
            coefficients = taylor_coefficients(V, G, max_order)
            partial_sums = np.cumsum(coefficients)
            for order in range(max_order + 1):
                series_records.append({
                    "N": int(size), "direction": f"{source}_to_{target}",
                    "order": order, "C": coefficients[order],
                    "partial_sum": partial_sums[order],
                    "G_full_reweighted": full,
                    "error_vs_full_reweighting": partial_sums[order] - full,
                })
    return (pd.DataFrame.from_records(records),
            pd.DataFrame.from_records(series_records), removal)


def main():
    args = parse_args()
    try:
        frame = energy.load_and_validate(args.input.resolve())
        if args.n_bootstrap < 100 or args.max_order < 4:
            raise ValueError("Use at least 100 bootstrap resamples and max order >= 4")
        result, series, removal = analyze(
            frame, args.temperature, args.n_bootstrap,
            np.random.default_rng(args.seed), args.max_order)
        # The explicit fourth-order formulas and the series division are
        # independent implementations; they must agree for orders 0-4.
        for row in result.itertuples(index=False):
            low = series.loc[(series["N"] == row.N) & (series["direction"] == row.direction)
                             & (series["order"] <= 4), "C"].to_numpy()
            explicit = np.array([getattr(row, f"C{k}") for k in range(5)])
            if not np.allclose(low, explicit, rtol=1e-8, atol=1e-15):
                raise ValueError(f"Series and explicit cumulants differ: {row.direction} N={row.N}")
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, float_format="%.17g")
        series_output = args.series_output.resolve()
        series.to_csv(series_output, index=False, float_format="%.17g")
        print(f"Saved order-{args.max_order} empirical series to {series_output}")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(result)} directions and sizes to {output}")
    print(f"Removed {int(removal['n_removed'].sum())} endpoint frames from "
          f"{len(removal)} original trajectories")
    columns = ["N", "direction", "G_order0", "G_order1", "G_order2",
               "G_order3", "G_order4", "G_full_reweighted", "G_direct_target"]
    print(result[columns].to_string(index=False, float_format=lambda x: f"{x:.9g}"))
    print("\n|C_k|/SE(C_k) from frame resampling:")
    print(result[["N", "direction"] + [f"C{k}_abs_over_se" for k in range(1, 5)]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
