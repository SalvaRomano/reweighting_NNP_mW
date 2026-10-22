#!/usr/bin/env python3
"""Independently check saved mixed cumulants against derivatives of F(lambda)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KB_EV_PER_K = 8.617333262145e-5
NAMES = ("kappa_VG", "kappa_VVG", "kappa_VVVG", "kappa_VVVVG")
# A derivative passes when |numerical - analytic| <= ATOL + RTOL*|analytic|.
# The relative part matters: kappa_VVVVG is only ~1e-6 at N=64, so an
# absolute tolerance alone (formerly 1e-7) would accept an 8% error.
ATOL = 1e-10
RTOL = 1e-3


def numerical_derivatives(V, G):
    """Differentiate an eight-degree local interpolant of stable F(lambda)."""
    # Constant shifts in V cancel from F; a constant shift in G changes only
    # F(0). Centering both greatly reduces fourth-derivative roundoff.
    V = V - V.mean()
    G = G - G.mean()
    scale = np.max(np.abs(V))
    step = 0.08 if scale == 0 else min(0.08, 0.4 / scale)
    x = np.arange(-4, 5, dtype=float)
    values = []
    for grid_point in x:
        logw = grid_point * step * V
        weights = np.exp(logw - logw.max())
        values.append(np.dot(weights, G) / weights.sum())
    coefficients = np.polynomial.polynomial.polyfit(x, values, 8)
    derivatives = [math.factorial(order) * coefficients[order] / step ** order
                   for order in range(1, 5)]
    return step, derivatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results/cumulant_expansion/summary.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cumulant_expansion/derivative_verification.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, usecols=[
        "N", "model", "source_run", "step", "poteng", "cross_poteng",
    ])
    grouped = frame.groupby(["model", "N", "source_run"])["step"]
    frame = frame.loc[(frame["step"] != grouped.transform("min"))
                      & (frame["step"] != grouped.transform("max"))]
    summary = pd.read_csv(args.summary)
    records = []
    for row in summary.itertuples(index=False):
        group = frame.loc[(frame["N"] == row.N) & (frame["model"] == row.source_model)]
        if len(group) != row.n_source_frames:
            raise ValueError(f"Frame count mismatch for {row.direction} N={row.N}")
        # Independently reconstruct V and G from the primary energy columns.
        V = -(group["cross_poteng"].to_numpy() - group["poteng"].to_numpy()) \
            / (KB_EV_PER_K * row.temperature_K)
        G = group["cross_poteng"].to_numpy() / row.N
        step, derivatives = numerical_derivatives(V, G)
        record = {"N": row.N, "direction": row.direction, "step_lambda": step}
        for name, numerical in zip(NAMES, derivatives):
            analytic = getattr(row, name)
            difference = abs(numerical - analytic)
            record[f"{name}_numerical"] = numerical
            record[f"{name}_analytic"] = analytic
            record[f"{name}_absolute_difference"] = difference
            record[f"{name}_relative_difference"] = difference / abs(analytic)
            if difference > ATOL + RTOL * abs(analytic):
                raise ValueError(f"Derivative check failed: {row.direction} N={row.N} {name}")
        records.append(record)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, float_format="%.17g")
    print(f"Verified four derivatives for {len(records)} groups; saved {output}")
    for kind in ("absolute", "relative"):
        largest = max(record[f"{name}_{kind}_difference"]
                      for record in records for name in NAMES)
        print(f"Largest {kind} derivative discrepancy: {largest:.3g}")
    print(f"Tolerance: |difference| <= {ATOL:g} + {RTOL:g}*|analytic|")


if __name__ == "__main__":
    main()
