# Corrected mW evaluation of saved NNP trajectories

This calculation evaluates the saved NNP configurations with the corrected
`lammps/mW_NVT/mW.sw` supplied on 2026-09-08. No dynamics are run. The exact
potential is preserved as `mW.sw`; `manifest.json` records its SHA256 checksum,
the source-table checksum, and the trajectories and final steps evaluated.

Each `N_*/run_*/` contains a generated initial configuration, a single `rerun`
input, and the LAMMPS log and screen output. The generated initial configuration
already has the full particle count, so the input does not replicate it.
Reruns stop at the last step retained in the existing source table, before any
later failed MD frames. All original trajectories, source energies, and previous
analysis outputs are preserved.

Analysis outputs:

- `nnp_frames_with_corrected_mw.csv`: retained source frames joined to new target
  energies by system size, original run, and step; original sample IDs retained.
- `summary_by_size.csv`: NNP-to-corrected-mW reweighting pooled across original
  runs at each N, including ESS and dominant-weight diagnostics.
- `summary_by_run.csv`: the same quantities separately for each original run.
- `endpoint_removal.csv`: frame selection counts for each trajectory.

The source-table frame selection and first/last-frame removal match the previous
analysis. Weights use total target-minus-source potential energy at 300 K;
reported mean energies are eV per particle. Source NNP energies are the original
MD log values; target energies are evaluated on the saved coordinates.

There is no direct corrected-mW average, reverse-direction result, or validation
error against a direct corrected-mW ensemble here. The old direct mW trajectory
was sampled using different SW parameters. Its average cannot serve as the
direct reference for this corrected potential. The prior concerns about NNP
trajectory failures and equilibration also remain separate from this correction.

Commands run from the production directory:

```bash
MPLCONFIGDIR=/tmp/reweighting_mpl python scripts/rerun_corrected_mw.py prepare --lammps /data/home/salvatorer26/n2p2-committee-modified/bin/lmp_mpi
bash results/corrected_mw_rerun/run_reruns.sh
MPLCONFIGDIR=/tmp/reweighting_mpl python scripts/rerun_corrected_mw.py analyze
```

Preparation refuses to overwrite an existing output directory. To prepare a
separate calculation, pass a new `--output-dir` and use that same option for
analysis. The launcher runs single-process LAMMPS jobs sequentially; the MPI
runtime requires local sockets. It contains only single-point reruns.

## Figures

Run `MPLCONFIGDIR=/tmp/reweighting_mpl python scripts/plot_corrected_mw.py`
to reproduce PNG and PDF plots under `figures/Reweighting/corrected_mw_rerun/`:
ESS fraction, largest weight fraction, top-10 weight fraction, and six
thermodynamic trace figures under `thermo_vs_configuration/`. These use the
same styling as the original analysis plots and show only the corrected
direction. Trace boundaries and endpoint removal use the original `source_run`
identifiers, not the pooled `run=0` label. The script validates trace frame
counts against the corrected summary before plotting.

The reweighting-error figure is omitted because no directly sampled
corrected-mW reference ensemble is available.
