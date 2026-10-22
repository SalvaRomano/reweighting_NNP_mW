#!/usr/bin/env python3
"""Analyze the N=64 mW potential-energy autocorrelation from a LAMMPS log."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MARKER = "BEGIN_AUTOCORRELATION_PRODUCTION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results/energy_autocorrelation_N64/log.lammps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/energy_autocorrelation_N64/autocorrelation.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results/energy_autocorrelation_N64/summary.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures/Autocorrelation/energy_autocorrelation_N64.png",
    )
    parser.add_argument("--max-lag-ps", type=float, default=500.0)
    parser.add_argument("--plot-max-lag-ps", type=float, default=100.0)
    parser.add_argument("--title", default="N=64 mW potential-energy autocorrelation")
    parser.add_argument("--blocks", type=int, default=8)
    return parser.parse_args()


def read_production_thermo(path: Path) -> pd.DataFrame:
    """Read the Step/Time/Temp/PotEng block following the production marker."""
    rows: list[tuple[int, float, float, float]] = []
    active = False
    in_table = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER in line:
                active = True
                in_table = False
                rows.clear()
                continue
            if not active:
                continue
            fields = line.split()
            if fields == ["Step", "Time", "Temp", "PotEng"]:
                in_table = True
                continue
            if not in_table or len(fields) != 4:
                continue
            try:
                step = int(fields[0])
                values = tuple(float(value) for value in fields[1:])
            except ValueError:
                continue
            rows.append((step, *values))
    frame = pd.DataFrame(rows, columns=["step", "time_ps", "temperature_K", "poteng_eV"])
    if frame.empty:
        raise ValueError(f"No production thermo rows found after {MARKER!r} in {path}")
    frame = frame.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    spacing = np.diff(frame["time_ps"].to_numpy())
    if len(spacing) == 0 or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-10):
        raise ValueError("Production energy samples are not uniformly spaced")
    return frame


def normalized_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the unbiased, variance-normalized autocorrelation by FFT."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    if centered.ndim != 1 or len(centered) < 2 or not np.isfinite(centered).all():
        raise ValueError("Energy series must be a finite one-dimensional vector")
    transform_length = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_length)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=transform_length)
    covariance = covariance[: max_lag + 1] / np.arange(len(centered), len(centered) - max_lag - 1, -1)
    if covariance[0] <= 0:
        raise ValueError("Energy series has zero variance")
    return covariance / covariance[0]


def crossing_time(lags: np.ndarray, acf: np.ndarray, threshold: float) -> float:
    """Linearly interpolate the first downward threshold crossing."""
    indices = np.flatnonzero(acf <= threshold)
    indices = indices[indices > 0]
    if not len(indices):
        return np.nan
    right = int(indices[0])
    left = right - 1
    fraction = (threshold - acf[left]) / (acf[right] - acf[left])
    return float(lags[left] + fraction * (lags[right] - lags[left]))


def integrated_time(dt: float, acf: np.ndarray) -> tuple[float, int]:
    """Integrate the initial positive sequence of the normalized ACF."""
    nonpositive = np.flatnonzero(acf[1:] <= 0)
    stop = int(nonpositive[0] + 1) if len(nonpositive) else len(acf) - 1
    tau = dt * (0.5 + float(np.sum(acf[1:stop])))
    return tau, stop


def block_acfs(values: np.ndarray, blocks: int, max_lag: int) -> np.ndarray:
    """Return equal-block ACFs for a between-block standard error."""
    if blocks < 2:
        raise ValueError("At least two blocks are required")
    block_length = len(values) // blocks
    if max_lag >= block_length:
        raise ValueError("Maximum lag must be shorter than each block")
    trimmed = values[: blocks * block_length].reshape(blocks, block_length)
    return np.stack([normalized_acf(block, max_lag) for block in trimmed])


def main() -> int:
    args = parse_args()
    frame = read_production_thermo(args.log.resolve())
    dt = float(frame.time_ps.iloc[1] - frame.time_ps.iloc[0])
    max_lag = min(int(np.floor(args.max_lag_ps / dt)), len(frame) // args.blocks - 1)
    values = frame.poteng_eV.to_numpy(dtype=float)
    acf = normalized_acf(values, max_lag)
    block_values = block_acfs(values, args.blocks, max_lag)
    acf_sem = np.std(block_values, axis=0, ddof=1) / np.sqrt(args.blocks)
    lags = np.arange(max_lag + 1) * dt
    tau_int, positive_stop = integrated_time(dt, acf)
    tau_exp = crossing_time(lags, acf, np.exp(-1.0))
    zero_crossing = crossing_time(lags, acf, 0.0)
    production_time = float(frame.time_ps.iloc[-1] - frame.time_ps.iloc[0])
    statistical_inefficiency = 2.0 * tau_int / dt
    effective_samples = len(frame) / statistical_inefficiency

    output = pd.DataFrame({"lag_ps": lags, "acf": acf, "acf_sem_blocks": acf_sem})
    summary = pd.DataFrame([{
        "n_samples": len(frame),
        "sample_interval_ps": dt,
        "production_time_ps": production_time,
        "mean_temperature_K": float(frame.temperature_K.mean()),
        "std_temperature_K": float(frame.temperature_K.std(ddof=1)),
        "mean_poteng_eV": float(np.mean(values)),
        "std_poteng_eV": float(np.std(values, ddof=1)),
        "acf_1_over_e_crossing_ps": tau_exp,
        "acf_first_zero_crossing_ps": zero_crossing,
        "integrated_autocorrelation_time_ps": tau_int,
        "integration_positive_sequence_stop_ps": positive_stop * dt,
        "statistical_inefficiency_samples": statistical_inefficiency,
        "effective_independent_samples": effective_samples,
        "n_blocks_for_acf_sem": args.blocks,
        "max_reported_lag_ps": lags[-1],
    }])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.12g")
    summary.to_csv(args.summary, index=False, float_format="%.12g")

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.plot(lags, acf, color="#2166ac", lw=1.5, label="Potential-energy ACF")
    ax.fill_between(lags, acf - acf_sem, acf + acf_sem, color="#2166ac", alpha=0.2,
                    label=f"Block SEM ({args.blocks} blocks)")
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.axhline(np.exp(-1.0), color="#d95f02", ls="--", lw=1.0, label="1/e")
    if np.isfinite(tau_exp):
        ax.axvline(tau_exp, color="#d95f02", ls=":", lw=1.2,
                   label=rf"1/e crossing = {tau_exp:.2f} ps")
    ax.set_xlabel("Lag (ps)")
    ax.set_ylabel("Normalized autocorrelation")
    ax.set_title(args.title)
    ax.set_xlim(0.0, min(args.plot_max_lag_ps, lags[-1]))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=300, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved {args.output}, {args.summary}, and {args.figure.with_suffix('.*')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
