# Configuration bootstrap of truncation residuals

Run from the production directory:

```bash
python scripts/bootstrap_cumulant_truncation.py
python -m unittest discover -s tests -p 'test_cumulant*.py'
```

The run uses 1,000 resamples per size/direction, seed 20261006 (the existing
cumulant-analysis convention), temperature 300 K, and orders 0–12. Each draw
samples M indices with replacement from the retained source ensemble. The same
indices select V and A, and both the complete centered Taylor calculation and
the shifted-exponential full mean are recomputed for that draw. No independent
coefficient-error propagation is used. Resampling pools the retained NNP runs,
as in the existing cumulant analysis; it is an ordinary frame bootstrap, not a
trajectory or block bootstrap.

## Reused code and validation

- `scripts/compute_energy_averages_and_reweighting.py`: `load_and_validate`,
  `remove_trajectory_endpoints`, and `KB_EV_PER_K`.
- `src/cumulant_expansion.py`: `prepare_observables`, `taylor_coefficients`,
  and `tilted_mean`. `contributions` independently checks orders 0–4 on a subset.
- `scripts/analyze_cumulant_expansion.py`: input path, ordering of size/direction
  groups, pooled-source endpoint convention, and seed 20261006.

The run removed 72 endpoint frames and reproduced all 156 saved original signed
residuals bit for bit against `results/cumulant_expansion/taylor_series_order12.csv`,
loaded with round-trip float parsing. Three bootstrap draws of four retained
N=64 NNP frames were checked against explicit order-0–4 formulas and a separately
evaluated exponential ratio. All eight cumulant/bootstrap unit tests passed;
these include fixed-index draws whose full means differ from the original full
mean, constant-frame draws through order 12, and separate signed/absolute quantiles.

Changing BLAS thread settings can change floating-point reductions: forcing one
thread produced differences of approximately 2.2e-16 eV/particle relative to the
saved reference, so the exact-reference run used the environment's default BLAS
settings. Such differences matter relatively only for near-machine-precision
residuals; no numerical floor is applied to CSV values.

## Outputs

- `results/cumulant_truncation_bootstrap.csv`: all 156 rows, with original signed
  and absolute residuals, signed bootstrap mean/median/sample standard deviation,
  signed 2.5/16/50/84/97.5 percentiles, absolute median and 2.5/16/84/97.5
  percentiles, frame counts, bootstrap count, seed, and temperature. All energy
  residuals are in eV/particle; percentiles use NumPy's default linear interpolation.
- `figures/Cumulant/appendix/truncation_error_bootstrap_large_N.{pdf,png}`:
  N=1024 (purple) and N=2048 (brown), circles for NNP → mW, squares for mW → NNP.
  Marked lines are original-sample residuals; darker bands show 16–84%, lighter
  bands show 2.5–97.5%. Display values alone are clipped at 1e-16 on the log axis.

Existing production outputs and source data were not overwritten. The script
prints the requested orders 4, 8, and 12 for both large sizes and directions.

## Interpretation

At N=2048, NNP → mW, order 12, the original residual is 5.378e-5 eV/particle,
the bootstrap median is 4.237e-5, the 16–84% interval is
[7.415e-6, 2.315e-4], and the 2.5–97.5% interval is [1.215e-6, 6.282e-4].
The 95% percentile range spans about 2.7 orders of magnitude and includes much
smaller residuals than the original value. Slow empirical convergence is therefore
not narrowly stable under resampling; its magnitude is highly sample-sensitive.
The bootstrap median nevertheless remains large compared with the N=1024
order-12 residuals.

N=1024 NNP → mW is also sensitive: at order 12 its original residual is
4.295e-9, while its bootstrap median is 4.743e-8 and its 95% percentile range
is [2.298e-9, 1.660e-7] eV/particle.

These are empirical percentile ranges for sensitivity to the observed sample,
not guarantees about population truncation error. Ordinary bootstrap cannot
represent configurations that were never sampled and does not correct temporal
correlation between frames.
