# Full mW trajectory reweighting

This run evaluates all 10,001 saved configurations in each of the six mW NVT
trajectories with the same NNP rerun settings used for the earlier 2,001-frame
analysis. The original NNP samples and their mW cross energies are reused.
The original 2,001-frame merged table is preserved here as
`thermo_all_with_cross_merged_first_2001.csv`. The validated full merged table
is now also installed at `analysis/thermo_all_with_cross_merged.csv`. The
earlier reweighting summary is preserved as
`energy_averages_and_reweighting_first_2001.csv`; the current analysis summary
and the full-trajectory summary use the same expanded mW data.

From the `production/` directory, the protocol is:

```bash
python scripts/full_mw_trajectory_reweighting.py prepare
bash results/full_mw_trajectory/run_reruns.sh /data/home/salvatorer26/n2p2-committee-modified/bin/lmp_mpi
python scripts/full_mw_trajectory_reweighting.py analyze
python scripts/plot_reweighting_diagnostics.py \
  --input results/full_mw_trajectory/energy_averages_and_reweighting.csv \
  --output-dir figures/Reweighting/full_mw_trajectory
```

`prepare` refuses to replace an existing output directory. The launcher runs
one LAMMPS process per size and writes its log in `N_<N>/`. The analysis checks
that each of the 10,001 dump steps has exactly one source thermo row and one
NNP rerun energy, that each rerun completed cleanly with the expected atom
count, and that all 2,001 previously evaluated frames agree with the earlier
cross energies. The maximum overlap difference is about 1.1e-13 eV.

The first and last saved frames are excluded, leaving 9,999 mW frames per size.
Weights use `exp[-(cross_poteng - poteng)/(kB * 300 K)]` with the **total**
potential-energy difference in eV. Only the averaged observable is divided by
N. The new merged table, summary, endpoint counts, overlap checks, and
old-versus-full comparison are CSV files in this directory.
The combined absolute-error figure is
`figures/Reweighting/full_mw_trajectory/absolute_error_before_after_both_directions_vs_N.{png,pdf}`.

The mW-to-NNP absolute after-error is smaller than in the 2,001-frame analysis
at all six sizes. At N=1024 it remains larger than the unweighted error. At
N=2048 the effective sample size falls from 94.2 to 59.5, and the largest
normalized weight rises from 4.64% to 11.0%; more frames therefore do not
remove the overlap concern. NNP-to-mW estimates use the unchanged NNP frames,
but their errors change because the direct mW reference now uses the full
trajectory.
