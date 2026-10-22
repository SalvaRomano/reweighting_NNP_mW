# Leading-order ratio-estimator bias versus system size

Run from `production/`:

```bash
python scripts/analyze_bias_vs_size.py
python scripts/plot_combined_bias.py --size 512
```

Only the leading `1/n` term of the self-normalized ratio-estimator bias is
used. For each N and direction, `summary.csv` is computed from the full source
dataset after removing the first and last frame of each original trajectory
(M frames). Weights use the total `cross_poteng - poteng` at 300 K; the
observable is `G = cross_poteng/N`.

Columns:

- `B_theory`: bias coefficient, `E[theta_hat_n] - theta ≈ B_theory/n`.
- `B_ci95_low`, `B_ci95_high`: 95% interval of `B_theory` from 2,000 resamples
  of the source frames (seed 20261006). This measures how well the finite
  dataset determines B. It is separate from the n-sample bootstrap in
  `analysis/bias_N*.csv`, which treats the dataset itself as the population
  and therefore only tests the 1/n expansion.
- `inefficiency_M_over_neff` and `delta_theta_w2 = theta_w2 - theta`: the two
  factors of the exact identity `B = -(M/N_eff)(theta_w2 - theta)`, where
  `theta_w2` averages G with squared weights.
- `sigma`, `se_at_M = sigma/sqrt(M)`: leading-order standard error of the
  ratio estimator, `sigma = sqrt(E[w^2 (G-theta)^2])/E[w]`, assuming
  independent frames.
- `bias_over_se_at_M`: `|B/M| / se_at_M`, the bias relative to the
  statistical error at the sample size actually used.

The 1/n expansion is reliable when `n >> M/N_eff`, i.e. at `n = M` when
`N_eff` is large. Figures mark `N_eff < 100` (both N=2048 directions) as
marginal.

Figures in `figures/Bias/vs_size/`:

- `bias_coefficient_vs_N`: |B_theory| with its 95% interval (open markers:
  the interval includes zero).
- `bias_decomposition_vs_N`: M/N_eff and |theta_w2 - theta| in two panels.
- `bias_to_noise_vs_N`: |bias|/SE at n = M, with the line bias = SE.

`figures/Bias/scaled_bias_N512_combined` compares the N=512 bootstrap `n*bias`
(n >= 100) with `B_theory` and its 95% band; it is the direct test of the
leading-order formula. At N=64 the bias is below the bootstrap resolution of
100,000 repetitions and is not used for this test.

Main results: B grows from ~6e-5 eV/particle at N=64 to ~0.1 at N=2048,
driven by M/N_eff (1.2 to 250); |theta_w2 - theta| stays between 1e-5 and
6e-4. At the actual sample sizes the bias is at most ~1% of the statistical
error for N <= 1024 and 9-31% at N=2048, where the expansion is marginal.
The 95% intervals of B are wide (roughly a factor of 2 at N=512) and include
zero for several sizes, so B is known in magnitude, not precisely.
