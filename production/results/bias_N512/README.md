# N=512 finite-sample ratio-estimator bias

Run from `production/`:

```bash
python scripts/analyze_ratio_bias.py --size 512 --min-fit-n 50
python scripts/plot_combined_bias.py --size 512
```

The calculation uses 100,000 bootstrap repetitions with seed 20260923 and
the N=512 source rows after removing the first and last saved frame of every
original trajectory, as in the reweighting and cumulant analyses: 15,086 NNP
frames and 9,999 mW frames. Definitions,
sample sizes, total-energy weights, fixed population weight shift, population
moments, and through-origin weighted fits match the N=64 analysis.

The full-source effective sample sizes are 4,669.63 for NNP-to-mW and 2,887.96
for mW-to-NNP, corresponding to fractions 0.3095 and 0.2888, matching
`analysis/energy_averages_and_reweighting.csv`. `B_theory` is 4.70e-4
(NNP-to-mW) and -6.20e-4 (mW-to-NNP) eV/particle. With the primary fit cutoff
n=50, `B_empirical/B_theory` is 0.785 for NNP-to-mW and 0.755 for mW-to-NNP;
with n>=100 it is 0.95 and 0.90. The cutoff-stability fits are in
`analysis/bias_N512_fits.csv`. The lower ratio at n>=50 is expected: the
next-order 1/n^2 term is still about 30-50% of the leading term at n=50.

Earlier versions of this analysis kept the trajectory endpoints. The identical,
unequilibrated step-0 frame shared by all five NNP runs (about 7 sigma from
the median G) then entered five times, which lowered NNP-to-mW `B_theory` to
1.70e-4 and made its fits unstable (0.39-2.0 x B_theory).
