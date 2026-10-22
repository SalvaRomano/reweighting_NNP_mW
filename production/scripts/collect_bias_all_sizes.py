#!/usr/bin/env python3
"""Combine per-size finite-sample bias outputs into all-size CSV tables."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 128, 256, 512, 1024, 2048)
TABLES = {
    "bias_all_sizes.csv": "bias_N{size}.csv",
    "bias_all_sizes_summary.csv": "bias_N{size}_summary.csv",
    "bias_all_sizes_fits.csv": "bias_N{size}_fits.csv",
}


def main() -> None:
    analysis_dir = ROOT / "analysis"
    for output_name, input_pattern in TABLES.items():
        frames = []
        for size in SIZES:
            path = analysis_dir / input_pattern.format(size=size)
            frame = pd.read_csv(path)
            frame.insert(0, "N", size)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        output = analysis_dir / output_name
        combined.to_csv(output, index=False, float_format="%.15g")
        print(f"Saved {output} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
