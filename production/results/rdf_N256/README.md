# N=256 direct and reweighted particle RDF

Run from `production/`:

```bash
python -m unittest discover -s tests -p 'test_rdf.py' -v
python scripts/compute_rdf_N256.py --preflight-only
python scripts/compute_rdf_N256.py
```

Input energies and frame keys come from
`analysis/thermo_all_with_cross_merged.csv`. Coordinates come from the full
`lammps/mW_NVT/N_256/water.lammpstrj` and the five
`lammps/NN_NVT/N_256/run_*/water.lammpstrj` dumps. Frames are joined by
`(model, source_run, step)` after excluding the first and last saved steps of
each original trajectory. There are 15,924 matched NNP frames and 9,999 matched
mW frames. Every retained master row has a trajectory match. The dumps have 95
unmatched NNP frames (10 excluded endpoints and 85 absent from the master)
and two excluded mW endpoints.

All frames have the same fixed cubic box: L = 19.7205782451 Å,
V = 7669.35667 Å³, and number density = 0.0333795925 Å⁻³. Thus the NNP and
mW NVT box volumes are compatible. The shared radial range ends at
L/2 = 9.86028912255 Å. Nominal bin width is 0.02 Å; the terminal bin ends
exactly at L/2. Its shell volume uses its actual edge spacing.

For each frame, a periodic cKDTree counts minimum-image pairs. Its ordered
pair counts are divided by two, and the unordered histogram is normalized by
`0.5 * N * (N/V_i) * (4*pi/3) * (r_outer**3 - r_inner**3)` before it enters
either average. The two reweighted curves use separately normalized frame
weights `exp[-beta * (cross_poteng - poteng)]` at 300 K, with the **total**
energy difference in the exponent. Effective sample sizes are 8361.0459 for
NNP to mW and 5134.5951 for mW to NNP.

The numerical curves are in `analysis/rdf_N256.csv`; the PNG and PDF figures
are `figures/Structural/rdf_N256.*`. A separate brute-force minimum-image histogram of
an actual frame matched the cKDTree implementation. The four curves average
about 0.99 over 8–9 Å, a useful normalization check.

## Residuals and first-shell features

Run `python scripts/analyze_rdf_N256.py` from `production/`. The signed
residuals are direct NNP minus direct mW, reweighted NNP-to-mW minus direct
mW, and reweighted mW-to-NNP minus direct NNP. Their numerical curves are in
`results/rdf_N256/residuals.csv`; the plot is `figures/Structural/rdf_N256_residuals.*`.

`results/rdf_N256/first_shell.csv` reports the first prominent peak, the
following prominent minimum, and coordination number for each of the four
RDFs. Features use the original unsmoothed curves, with peak prominence 0.05
and minimum prominence 0.03. Peak and minimum positions are reported for each
curve, but all four coordination numbers use one common cutoff
`r_cut_A = 3.50 Å` (`COMMON_CUTOFF_A` in `src/rdf_features.py`), the same
cutoff used for every size in `results/rdf_all_sizes/coordination_summary.csv`. The minimum is shallow, and
a one-bin (0.02 Å) shift of a per-curve cutoff changes coordination by about
0.08, which would swamp the reweighting differences. Coordination is
`4πρ ∫₀^r_cut g(r)r²dr`, evaluated with each RDF bin constant over its radial
shell. The density comes from the fixed N=256 trajectory box.
