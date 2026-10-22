# N=64 finite-sample ratio-estimator bias

Run from `production/`:

```bash
python -m unittest discover -s tests -p 'test_ratio_bias.py' -v
python scripts/analyze_ratio_bias.py --size 64 --min-fit-n 50
python scripts/plot_combined_bias.py --size 64
```

The calculation uses the N=64 source rows in
`analysis/thermo_all_with_cross_merged.csv` after removing the first and last
saved frame of every original trajectory, as in the reweighting and cumulant
analyses: 26,244 NNP frames and 9,999 mW frames. It uses the nominal NVT temperature of 300 K. For each direction,
`delta_U = cross_poteng - poteng` is the total energy difference and
`G = cross_poteng / 64` is the target potential energy per particle. A single
shift `max(logw)` defines the population `X = G*w` and `Y = w` arrays before
any bootstrap sampling. The direct target ensemble is never used in the bias
definition.

`analysis/bias_N64.csv` contains the 100,000-repetition bootstrap results and
`analysis/bias_N64_summary.csv` contains the full-source moments. Population
variance and covariance use `ddof=0`. Figures are named
`figures/Bias/bias_N64_<direction>.*` and
`figures/Bias/scaled_bias_N64_<direction>.*`.

`analysis/bias_N64_fits.csv` contains through-origin weighted least-squares
fits of `bias = B_empirical/n` at lower cutoffs n=50, 100, 200, and 500. The
weights are `1/SEM²`. The configurable primary cutoff defaults to n=200.
The latest production run used `--min-fit-n 50`, so its primary reported fit
includes every tested sample size.

The theoretical biases remain smaller than the Monte Carlo uncertainty in
the mean of 100,000 ordinary bootstrap estimates. Consequently, empirical
biases change sign across n and the fitted coefficients vary with the lower
cutoff. The fit standard errors are larger than the theoretical coefficients,
so this calculation does not resolve the predicted 1/n term precisely.
