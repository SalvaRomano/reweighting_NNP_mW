# RDF and coordination analysis across system sizes

Run from `production/`:

```bash
for n in 64 128 512 1024 2048; do
    python scripts/compute_rdf_N256.py --size "$n" --no-plot
done
python scripts/summarize_rdf_all_sizes.py
```

The existing `analysis/rdf_N256.csv` and `figures/Structural/rdf_N256.png/.pdf` remain
the N=256 numerical RDF and representative figure. The five other per-size
curves are `results/rdf_all_sizes/rdf_N*.csv`. Each curve comes from a
frame-normalized periodic pair histogram. Trajectories are matched to
`analysis/thermo_all_with_cross_merged.csv` by model, original run, and LAMMPS
timestep; the first and last step of each original run are excluded. Each
reweighting direction uses separately normalized weights based on the **total**
cross minus source potential energy at 300 K.

All four coordination numbers at every size integrate their RDFs to one
common radius, `r_cut = 3.50 Å` (`COMMON_CUTOFF_A` in `src/rdf_features.py`,
overridable with `--cutoff`), using `4πρ ∫₀^r_cut g(r)r²dr`. Bins are treated
as constant over their true shell volumes, with the partial volume of the
cutoff bin. The value is the mean of the six per-size direct mW first minima
(reported as `r_min_mW`: 3.49-3.53 Å). Those minima are set by bin-level noise
in a very flat minimum, and a 0.01 Å shift changes coordination by about 0.04,
so a per-size cutoff produced a spurious N-trend. First peaks and the mW first
minimum are detected on the unsmoothed curves with respective prominence
thresholds of 0.05 and 0.03. The number density (`number_density_A3`) is read
from the common fixed NVT box for each size.

At the common cutoff, coordination depends on the box density, not on N. The
NVT boxes of N=512, 1024, and 2048 are 2x2x2 replicas of N=64, 128, and 256,
so the sizes form three density pairs: 64/512 at 0.979, 128/1024 at 0.967, and
256/2048 at 0.999 g/cm³. Coordination is about 5.03, 4.97, and 5.13 for these
pairs, and members of a pair agree to about 0.002. The x-axis labels of
`rdf_coordination_all_sizes` show each box density. Before/after errors are
computed within one size and are unaffected by the density difference.

`coordination_summary.csv` contains the requested peaks, cutoff,
coordination numbers, and absolute before/after errors. Figures are
`figures/Structural/rdf_coordination_all_sizes.png/.pdf` and
`figures/Structural/rdf_coordination_errors_all_sizes.png/.pdf`. The two before-error
columns are numerically identical by definition; they appear separately in
the error figure to preserve the two directional comparisons.

At N=2048, the matched ensembles contain 2,276 NNP and 9,999 mW frames, but
the effective sample sizes are only 9.11 for NNP-to-mW and 59.50 for
mW-to-NNP. The NNP-to-mW coordination error increases after reweighting at
this size; treat that weighted estimate as sensitive to a few configurations.

## Signed coordination differences with errors

Run from `production/` (per-frame values are cached; add `--recompute` to
re-read the trajectories):

```bash
python scripts/analyze_coordination_differences.py
```

For every matched frame (same matching, endpoint exclusion, and 300 K
total-energy weights as the RDFs), the number of neighbours within
`r_cut = 3.50 Å` is counted directly (`src/coordination.py`). The frame means
reproduce the RDF-integrated coordination numbers above to 5e-8. Within each
size the script reports `CN(NNP) − CN(mW)`, `CN(NNP→mW) − CN(mW)`, and
`CN(mW→NNP) − CN(NNP)`. These signed differences cancel the box-density
dependence of the absolute values. Errors are block-bootstrap standard errors
(contiguous 50-frame blocks within each run, NNP and mW resampled
independently, weights renormalized in every replicate, 2000 replicates,
seed 20261013). Saved frames are nearly uncorrelated: block-averaged errors
stay within about 1.0–1.5× the naive error up to 100-frame blocks.

Outputs: `coordination_per_frame.csv`, `coordination_differences.csv`
(differences, standard errors, z-scores, N_eff), and
`figures/Structural/coordination_differences_all_sizes.png/.pdf`. The
reweighted N=2048 estimates have N_eff = 9 (NNP→mW) and 59 (mW→NNP); their
large error bars reflect this.
