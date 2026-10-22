#!/usr/bin/env python3
"""Compute direct and frame-reweighted particle RDFs from LAMMPS dumps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rdf import frame_rdf, radial_edges  # noqa: E402


N = 256
DR = 0.02
KB_EV_PER_K = 8.617333262145e-5
REQUIRED_COLUMNS = {"model", "N", "source_run", "step", "poteng", "cross_poteng"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256,
                        choices=(64, 128, 256, 512, 1024, 2048))
    parser.add_argument("--input", type=Path,
                        default=ROOT / "analysis/thermo_all_with_cross_merged.csv")
    parser.add_argument("--trajectory-root", type=Path, default=ROOT / "lammps")
    parser.add_argument("--output", type=Path,
                        help="Output CSV (default: existing N=256 path or results/rdf_all_sizes)")
    parser.add_argument("--figure-prefix", type=Path,
                        help="Figure prefix (default: existing N=256 path)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Only save numerical RDF curves")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="Nominal NVT temperature in K (default: 300)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check box compatibility and timestep coverage without RDF computation")
    return parser.parse_args()


def read_dump(path, positions=False):
    """Yield (step, atom count, lower bounds, lengths, coordinates) per frame."""
    with path.open(encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Unexpected dump marker in {path}: {marker.strip()}")
            step = int(handle.readline().strip())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Missing atom-count header in {path} at step {step}")
            count = int(handle.readline().strip())
            box_header = handle.readline().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"] or box_header[3:] != ["pp"] * 3:
                raise ValueError(f"Expected orthorhombic periodic box in {path} at step {step}")
            bounds = []
            for _ in range(3):
                fields = handle.readline().split()
                if len(fields) != 2:
                    raise ValueError(f"Invalid box bounds in {path} at step {step}")
                bounds.append((float(fields[0]), float(fields[1])))
            lower = np.array([pair[0] for pair in bounds])
            lengths = np.array([pair[1] - pair[0] for pair in bounds])
            header = handle.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"Missing atom-coordinate header in {path} at step {step}")
            columns = header[2:]
            if not {"id", "type", "x", "y", "z"}.issubset(columns):
                raise ValueError(f"Expected id/type/x/y/z columns in {path} at step {step}")
            coordinates = None
            if positions:
                xyz = [columns.index(axis) for axis in "xyz"]
                id_column = columns.index("id")
                type_column = columns.index("type")
                coordinates = np.empty((count, 3), dtype=float)
                atom_ids = set()
                for index in range(count):
                    fields = handle.readline().split()
                    if len(fields) != len(columns):
                        raise ValueError(f"Truncated atom data in {path} at step {step}")
                    atom_ids.add(int(fields[id_column]))
                    if int(fields[type_column]) != 1:
                        raise ValueError(f"Unexpected atom type in {path} at step {step}")
                    coordinates[index] = [float(fields[column]) for column in xyz]
                if len(atom_ids) != count:
                    raise ValueError(f"Duplicate atom IDs in {path} at step {step}")
            else:
                for _ in range(count):
                    if not handle.readline():
                        raise ValueError(f"Truncated atom data in {path} at step {step}")
            yield step, count, lower, lengths, coordinates


def load_master(path, temperature):
    """Select N=256 and remove each original trajectory's two endpoints."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing master columns: {', '.join(missing)}")
    frame = frame.loc[frame["N"] == N, list(REQUIRED_COLUMNS)].copy()
    for column in ("source_run", "step", "poteng", "cross_poteng"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {column} in N={N} master rows")
    if set(frame["model"]) != {"NNP", "mW"}:
        raise ValueError("Both NNP and mW N=256 ensembles are required")
    if frame.duplicated(["model", "source_run", "step"]).any():
        raise ValueError("Duplicate model/source_run/step keys in master")
    group_columns = ["model", "source_run"]
    counts = frame.groupby(group_columns).size()
    if (counts < 3).any():
        raise ValueError("Every source trajectory needs at least three frames")
    grouped_steps = frame.groupby(group_columns)["step"]
    retained = frame.loc[(frame["step"] != grouped_steps.transform("min"))
                         & (frame["step"] != grouped_steps.transform("max"))].copy()
    beta = 1.0 / (KB_EV_PER_K * temperature)
    weight_maps = {}
    neff = {}
    for model in ("NNP", "mW"):
        group = retained.loc[retained["model"] == model]
        # The exponent uses total energy in eV, while RDF normalization is per frame.
        delta_U = group["cross_poteng"].to_numpy() - group["poteng"].to_numpy()
        logw = -beta * delta_U
        weights = np.exp(logw - logw.max())
        probabilities = weights / weights.sum()
        weight_maps[model] = {
            (int(run), int(step)): float(probability)
            for run, step, probability in zip(group["source_run"], group["step"], probabilities)
        }
        neff[model] = float(1.0 / np.dot(probabilities, probabilities))
    return frame, retained, weight_maps, neff


def trajectory_paths(root):
    paths = [("mW", 1, root / f"mW_NVT/N_{N}/water.lammpstrj")]
    paths.extend(("NNP", run, root / f"NN_NVT/N_{N}/run_{run}/water.lammpstrj")
                 for run in range(1, 6))
    for _, _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def preflight(paths, master, retained):
    """Scan every box and timestep before any RDF calculation."""
    retained_keys = {
        (row.model, int(row.source_run), int(row.step))
        for row in retained.itertuples(index=False)
    }
    master_keys = {
        (row.model, int(row.source_run), int(row.step))
        for row in master.itertuples(index=False)
    }
    seen = set()
    stats = {}
    for model, run, path in paths:
        count = 0
        min_lengths = np.full(3, np.inf)
        max_lengths = np.zeros(3)
        min_volume, max_volume = np.inf, 0.0
        for step, atoms, _, lengths, _ in read_dump(path):
            key = (model, run, step)
            if key in seen:
                raise ValueError(f"Duplicate trajectory frame: {key}")
            seen.add(key)
            if atoms != N or not np.isfinite(lengths).all() or (lengths <= 0).any():
                raise ValueError(f"Invalid atom count or box at {key}")
            volume = np.prod(lengths)
            min_lengths = np.minimum(min_lengths, lengths)
            max_lengths = np.maximum(max_lengths, lengths)
            min_volume = min(min_volume, volume)
            max_volume = max(max_volume, volume)
            count += 1
        stats[(model, run)] = (count, min_lengths, max_lengths, min_volume, max_volume)
    missing_retained = retained_keys - seen
    if missing_retained:
        raise ValueError(f"{len(missing_retained)} retained master frames lack trajectory coordinates")
    global_min_length = min(np.min(value[1]) for value in stats.values())
    all_lengths = np.array([value[1] for value in stats.values()] +
                           [value[2] for value in stats.values()])
    compatible = np.allclose(all_lengths, all_lengths[0], rtol=0, atol=1e-6)
    print(f"N={N} box and frame preflight:")
    for (model, run), (count, minimum, maximum, vmin, vmax) in stats.items():
        matched = sum((model, run, step) in retained_keys
                      for m, r, step in seen if m == model and r == run)
        unmatched = count - matched
        other_unmatched = sum((m, r, step) not in master_keys
                              for m, r, step in seen if m == model and r == run)
        print(f"  {model} run {run}: dump={count}, retained matches={matched}, "
              f"unmatched dump={unmatched} ({other_unmatched} absent from master), "
              f"L_min={minimum}, L_max={maximum} Å, "
              f"V={vmin:.9g}..{vmax:.9g} Å³, "
              f"rho={N/vmax:.9g}..{N/vmin:.9g} Å^-3")
    if compatible:
        print("Box compatibility: NNP and mW lengths/volumes agree within 1e-6 Å.")
    else:
        print("WARNING: NNP and mW box lengths/volumes differ; interpret reweighting with care.")
    print(f"Smallest box length = {global_min_length:.10g} Å; r_max = "
          f"{global_min_length / 2:.10g} Å")
    print(f"Retained master frames without trajectory match: {len(missing_retained)}")
    return global_min_length / 2, retained_keys, compatible


def compute_curves(paths, weight_maps, edges, retained_keys):
    """Accumulate direct and normalized weighted frame RDFs independently."""
    sums = {model: np.zeros(len(edges) - 1) for model in ("NNP", "mW")}
    weighted = {model: np.zeros(len(edges) - 1) for model in ("NNP", "mW")}
    matched = {model: 0 for model in ("NNP", "mW")}
    unmatched = {model: 0 for model in ("NNP", "mW")}
    for model, run, path in paths:
        for step, count, lower, lengths, coordinates in read_dump(path, positions=True):
            key = (model, run, step)
            if key not in retained_keys:
                unmatched[model] += 1
                continue
            if count != N:
                raise ValueError(f"Unexpected particle count at {key}")
            p = weight_maps[model][(run, step)]
            g_frame = frame_rdf(coordinates, lower, lengths, edges)
            sums[model] += g_frame
            weighted[model] += p * g_frame
            matched[model] += 1
    for model in ("NNP", "mW"):
        if matched[model] != len(weight_maps[model]):
            raise ValueError(f"Missing matched frames during RDF calculation for {model}")
    return sums, weighted, matched, unmatched


def save_outputs(edges, sums, weighted, matched, output, figure_prefix, compatible):
    """Write numerical curves and the four-curve comparison figure."""
    radii = (edges[:-1] + edges[1:]) / 2
    curves = pd.DataFrame({
        "r": radii,
        "g_NNP": sums["NNP"] / matched["NNP"],
        "g_mW": sums["mW"] / matched["mW"],
        "g_NNP_to_mW": weighted["NNP"],
        "g_mW_to_NNP": weighted["mW"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output, index=False, float_format="%.12g")
    print(f"Saved {len(curves)} radial bins to {output}")

    if figure_prefix is None:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    styles = (
        ("g_NNP", "NNP direct", "tab:blue", "-"),
        ("g_mW", "mW direct", "tab:orange", "-"),
        ("g_NNP_to_mW", "NNP → mW reweighted", "tab:blue", "--"),
        ("g_mW_to_NNP", "mW → NNP reweighted", "tab:orange", "--"),
    )
    for column, label, color, line_style in styles:
        ax.plot(radii, curves[column], color=color, linestyle=line_style,
                linewidth=1.8, label=label)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    title = f"Particle-particle RDF, N = {N}"
    if not compatible:
        title += " (unequal box volumes)"
    ax.set_title(title)
    fig.tight_layout()
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = figure_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    global N
    args = parse_args()
    N = args.size
    output = args.output
    if output is None:
        output = (ROOT / "analysis/rdf_N256.csv" if N == 256
                  else ROOT / f"results/rdf_all_sizes/rdf_N{N}.csv")
    figure_prefix = args.figure_prefix
    if not args.no_plot and figure_prefix is None:
        figure_prefix = (ROOT / "figures/Structural/rdf_N256" if N == 256
                         else ROOT / f"figures/Structural/rdf_N{N}")
    try:
        master, retained, weight_maps, neff = load_master(args.input.resolve(),
                                                            args.temperature)
        paths = trajectory_paths(args.trajectory_root.resolve())
        r_max, retained_keys, compatible = preflight(paths, master, retained)
        if args.preflight_only:
            return 0
        edges = radial_edges(DR, r_max)
        sums, weighted, matched, unmatched = compute_curves(
            paths, weight_maps, edges, retained_keys)
        print(f"Matched retained frames: NNP={matched['NNP']}, mW={matched['mW']}")
        print(f"Unmatched trajectory frames: NNP={unmatched['NNP']}, mW={unmatched['mW']}")
        print(f"N_eff NNP → mW = {neff['NNP']:.8g}; "
              f"N_eff mW → NNP = {neff['mW']:.8g}")
        save_outputs(edges, sums, weighted, matched, output.resolve(),
                     figure_prefix.resolve() if figure_prefix is not None else None,
                     compatible)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
