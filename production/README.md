# Exponential reweighting between mW and NNP water models

## Project overview

This project studies practical limitations of exponential reweighting between
the mW water Hamiltonian and a neural-network potential (NNP). It covers
bidirectional energy reweighting, loss of overlap with system size,
fourth-order cumulant expansions, structural observables, and finite-sample
bias of the self-normalized ratio estimator.

The active processed data and figures cover `N = 64, 128, 256, 512, 1024,
2048`. Raw trajectories remain local for reproducibility but are ignored by
Git because they are too large for a normal GitHub repository.

## Requirements

Use Python 3 with NumPy, pandas, SciPy, and Matplotlib:

```bash
python -m pip install numpy pandas scipy matplotlib
```

Run commands from the repository root. LAMMPS is needed only to repeat the
cross-Hamiltonian reruns; current processed CSV files suffice for statistical
plots.

## Repository structure

```text
.
├── analysis/       # Current processed tables and compact summaries
├── figures/        # Current publication and diagnostic figures
├── lammps/         # Simulation inputs plus local raw NVT and rerun data
├── results/        # Specialized compact outputs and workflow notes
├── scripts/        # Preparation, reweighting, cumulant, RDF, and bias tools
├── src/            # Reusable numerical implementations
└── tests/          # Unit tests for cumulants, RDFs, and ratio bias
```

Archived legacy material is documented in
`archive/legacy_2026-09-23/MANIFEST.md`; `archive/` is ignored by Git.

## Simulation setup

The source ensembles are NVT simulations of single-species water at nominal
`T = 300 K` for all six sizes.

- mW uses one production trajectory per size.
- NNP uses multiple independent runs where available under
  `lammps/NN_NVT/N_<N>/run_<run>/`.
- Saved configurations are separated by approximately 50 ps. Earlier energy
  autocorrelation analysis found decorrelation at about 50 ps, so saved frames
  are treated as approximately independent in the bootstrap analysis.
- Each saved configuration is evaluated under the other Hamiltonian through a
  LAMMPS rerun. Frames are joined by timestep and original run identity, never
  by row position.

mW trajectories are under `lammps/mW_NVT/N_<N>/`; NNP trajectories are under
`lammps/NN_NVT/N_<N>/run_<run>/`. Corrected NNP-to-mW cross evaluations used by
the active dataset workflow are under `results/corrected_mw_rerun/`.

Some retained generated LAMMPS job and rerun files record the absolute paths
used on the original cluster. Regenerate portable rerun inputs with the active
preparation scripts, or edit executable, potential, and trajectory paths for
the target machine before launching LAMMPS.

## Cross evaluation

For a source configuration,

```text
delta_U = U_target - U_source
log(w)  = -beta * delta_U
beta    = 1 / (kB*T)
```

`delta_U` is the **total** potential-energy difference. It is never divided by
`N` in the exponential. Stable normalized weights use

```text
logw = logw - max(logw)
w_i  = exp(logw_i)
p_i  = w_i / sum_j w_j
```

with one fixed shift per full source dataset.

## Main merged dataset

The current master dataset is
`analysis/thermo_all_with_cross_merged.csv`. It contains the full mW trajectory
(10,001 saved frames per size) and pooled NNP runs. Important columns are:

- `model`: source ensemble (`NNP` or `mW`).
- `N`: particle count.
- `run`, `source_run`: pooled label and original trajectory identity.
- `step`: LAMMPS timestep used for matching.
- `temperature`: instantaneous source-run temperature.
- `poteng`: total source-Hamiltonian potential energy.
- `cross_poteng`: total target-Hamiltonian energy on the same configuration.
- `delta_U`: total `cross_poteng - poteng`.
- `delta_u`: per-particle energy difference.
- `volume`, `density`: source box observables when present.

The preparation stages are:

```bash
python scripts/collect_lammps_nvt.py --project-root lammps --output-dir analysis
python scripts/add_cross_poteng_to_thermo.py
python scripts/merge_nnp_runs.py --overwrite
```

Both preparation scripts read the mW-to-NNP cross energies from the
full-trajectory reruns in `results/full_mw_trajectory/N_<N>/` (option
`--mw-rerun-root`); NNP-to-mW energies come from `results/corrected_mw_rerun/`.
These commands reproduce the current master table. `merge_nnp_runs.py` refuses
to replace an existing master unless `--overwrite` is given.

The full-mW update is implemented by
`scripts/full_mw_trajectory_reweighting.py` and documented in
`results/full_mw_trajectory/README.md`. Do not replace the current merged table
with the earlier 2,001-frame collection.

## Reweighting analysis

```bash
python scripts/compute_energy_averages_and_reweighting.py
python scripts/plot_reweighting_diagnostics.py
python scripts/plot_thermo_vs_configuration.py
```

The main summary is `analysis/energy_averages_and_reweighting.csv`. Direct
cross averages, reweighted target averages, and signed/absolute before and
after errors are energies per particle and are not divided by `N` again.

For normalized weights, the effective sample size is

```text
N_eff = 1 / sum_i p_i^2.
```

Figures under `figures/reweighting/` show effective-sample fractions, maximum
and top-ten weight concentration, energy errors, and thermo traces. Overlap
decreases strongly with increasing system size.

## Cumulant expansion

The implementation uses `G = cross_poteng/N` and
`V = -beta*(cross_poteng-poteng)`, with the total energy difference in `V`.
Explicit mixed cumulants and partial sums are evaluated through fourth order.

```bash
python scripts/analyze_cumulant_expansion.py
python scripts/verify_cumulant_derivatives.py
python scripts/plot_cumulant_expansion.py
```

Current numerical results are in `results/cumulant_expansion/summary.csv` and
`derivative_verification.csv`. Tests and numerical derivatives of the stably
shifted tilted mean independently verify the formulas. Figures under
`figures/cumulant_expansion/` show truncation errors and contribution
magnitudes versus order for both directions.

## Structural analysis

RDFs are computed frame by frame with periodic minimum-image pair counts. Each
frame is normalized before ordinary or weighted averaging; an averaged RDF is
never reweighted. Bins use `dr = 0.02 Å` and extend to half the shortest box.

```bash
python scripts/compute_rdf_N256.py --size 256
python scripts/analyze_rdf_N256.py
python scripts/summarize_rdf_all_sizes.py
python scripts/plot_rdf_selected_sizes.py --sizes 1024 2048
```

Despite its historical filename, `compute_rdf_N256.py` accepts every supported
size through `--size`. Existing non-256 curves are in
`results/rdf_all_sizes/`; the representative N=256 curve is
`analysis/rdf_N256.csv`.

For each size, the first minimum of the direct mW RDF defines one common
coordination cutoff for direct NNP, direct mW, NNP-to-mW, and mW-to-NNP. The
summary is `results/rdf_all_sizes/coordination_summary.csv`. Current figures
include N=256, N=1024, and N=2048 RDFs, N=256 residuals, coordination versus
size, and before/after structural error.

## Finite-sample ratio bias

For target energy per particle `G = cross_poteng/N`, define

```text
X = G*w
Y = w
theta_hat_n = mean(X) / mean(Y).
```

The empirical reference `theta_full` uses all source frames; it is not a direct
target average. As in every other analysis, the first and last saved frame of each original
trajectory are removed first. Bootstrap sampling with replacement estimates

```text
bias_empirical(n) = mean_bootstrap(theta_hat_n) - theta_full.
```

The leading ratio-estimator bias is

```text
Bias(theta_hat_n) ≈ B_theory / n

B_theory = mu_X*Var(Y)/mu_Y^3 - Cov(X,Y)/mu_Y^2.
```

`B_theory` is computed once from full-source population moments with `ddof=0`.
`B_empirical` is fitted from `bias_empirical(n) = B_empirical/n`, with zero
intercept and weights `1/SEM²`.

Current 100,000-repetition analyses exist for N=64 and N=512:

```bash
python scripts/analyze_ratio_bias.py --size 64 --min-fit-n 50
python scripts/analyze_ratio_bias.py --size 512 --min-fit-n 50
python scripts/plot_combined_bias.py --size 64
python scripts/plot_combined_bias.py --size 512
python scripts/report_bias_fit_comparison.py --sizes 64 512
```

At N=64 the theoretical bias is very small compared with bootstrap-mean
fluctuations. At N=512 the bias is measurable in both directions and agrees
with the predicted `1/n` coefficient: `B_empirical/B_theory` is 0.95 and 0.90
for n>=100, and about 0.76-0.79 for n>=50, where the next-order `1/n^2` term
is still about 30-50% of the leading term.

## Reproducing current figures

| Figure family | Current script | Main input |
|---|---|---|
| Reweighting overlap and errors | `scripts/plot_reweighting_diagnostics.py` | `analysis/energy_averages_and_reweighting.csv` |
| Thermo traces | `scripts/plot_thermo_vs_configuration.py` | Master merged CSV |
| Cumulant truncation/contributions | `scripts/plot_cumulant_expansion.py` | `results/cumulant_expansion/summary.csv` |
| N=256 RDF | `scripts/compute_rdf_N256.py --size 256` | Master CSV and trajectories |
| N=256 RDF residuals | `scripts/analyze_rdf_N256.py` | `analysis/rdf_N256.csv` |
| Coordination and structural error | `scripts/summarize_rdf_all_sizes.py` | Per-size RDF CSVs |
| N=1024/N=2048 RDFs | `scripts/plot_rdf_selected_sizes.py --sizes 1024 2048` | Per-size RDF CSVs |
| Bias and scaled bias | `scripts/analyze_ratio_bias.py --size <N>` | Master merged CSV |
| Combined directional bias | `scripts/plot_combined_bias.py --size <N>` | `analysis/bias_N<N>.csv` |

## Validation

Run lightweight tests with:

```bash
python -m unittest discover -s tests -v
```

They cover periodic RDF normalization, common-cutoff coordination, explicit
mixed cumulants, tilted-mean derivatives, fixed weight shifts, population ratio
moments, and zero-intercept weighted bias fits.
