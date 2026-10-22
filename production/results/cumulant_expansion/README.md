# Cumulant expansion of exponential reweighting

Input: the full-mW version of `analysis/thermo_all_with_cross_merged.csv`.
The earlier 2,001-frame cumulant output is preserved locally under
`archive/legacy_2026-09-23/data/first_2001/`. The calculation uses the
same first-and-last saved-step exclusion per original trajectory as the existing
energy summary: 72 frames removed from 36 trajectories. Each model and N is
analyzed separately. The nominal NVT temperature is 300 K, giving one
`beta = 1/(kB T)` per group; instantaneous thermo temperatures are not used as
separate ensembles. Each mW source group has 9,999 retained frames. All
reported G values and contributions are eV/particle.

Run from `production/`:

```bash
python -m unittest discover -s tests -p 'test_cumulant_expansion.py' -v
python scripts/analyze_cumulant_expansion.py
python scripts/verify_cumulant_derivatives.py
python scripts/plot_cumulant_expansion.py
```

## Definitions

For each source frame, `delta_U = cross_poteng - poteng` is the **total**
potential-energy difference in eV, `V = -beta * delta_U`, and
`G = cross_poteng/N`. The target estimate is the tilted mean
`F(lambda) = <G exp(lambda V)> / <exp(lambda V)>` at `lambda = 1`. Its Taylor
coefficients at `lambda = 0` are `C_k = kappa(V,...,V,G)/k!` (k copies of V).
With `dV = V - mean(V)` and `dG = G - mean(G)`:

```text
C0 = mean(G)
C1 = mean(dV*dG)
C2 = mean(dV**2*dG) / 2
C3 = [mean(dV**3*dG) - 3*mean(dV**2)*mean(dV*dG)] / 6
C4 = [mean(dV**4*dG) - 6*mean(dV**2)*mean(dV**2*dG)
      - 4*mean(dV**3)*mean(dV*dG)] / 24
G_order_m = sum(C0, ..., Cm)
```

The moments are plain sample means, so `C_k` are the exact derivatives of the
sample's own `F(lambda)`. `G_full_reweighted` is `F(1)` on the same frames,
computed with shifted exponentials.

## Outputs

- `summary.csv`: per N and direction, the raw mixed cumulants, C0-C4, partial
  sums through order 4, `G_full_reweighted`, the direct target mean, and the
  errors of each partial sum against both references. It also has
  `C{k}_se` (bootstrap standard error) and `C{k}_abs_over_se` (|C_k|/SE) for
  k = 1-4, from 1,000 resamples of the source frames (seed 20261006,
  `n_bootstrap` column). These are the statistical uncertainties of C_k as
  estimates of the population cumulants.
- `taylor_series_order12.csv`: the empirical series through order 12 for all
  N and both directions: `C`, `partial_sum`, `G_full_reweighted`, and
  `error_vs_full_reweighting = partial_sum - G_full_reweighted`. The
  coefficients come from a power-series division of `<dG exp(lambda dV)>` by
  `<exp(lambda dV)>` (`taylor_coefficients` in `src/cumulant_expansion.py`).
  This is independent of the explicit formulas above; orders 0-4 agree to
  better than 1e-8 relative (checked on every run). Orders 1-12 at N=1024 and
  N=2048 agree with a 50-digit calculation to about 5e-14 relative.
- `derivative_verification.csv`: numerical first-to-fourth derivatives of
  `F(lambda)` from an eighth-degree local interpolant on a symmetric lambda
  grid, compared with the analytic cumulants for all 12 groups. A check
  passes when `|difference| <= 1e-10 + 1e-3*|analytic|`. The largest measured
  discrepancies are 4.2e-9 absolute and 4.2e-5 relative (both kappa_VVVVG,
  N=1024 NNP to mW).

Figures in `figures/Cumulant/`:

- `cumulant_coefficients_signed_bootstrap.*`: signed C1-C4 versus N with
  ±1 SE error bars, one row per direction. Filled markers have
  |C_k|/SE >= 3 (resolved); open markers are not resolved.
- `truncation_error_<direction>_order12.*`: |G_m - G_full| for m = 0-12.

The older `contribution_magnitude_*_vs_order.*` (|C_k| without errors) and
`truncation_error_*_vs_order.*` (orders 0-4) figures are superseded by these
and are no longer generated.

## What the truncation error measures

`|G_m - G_full|` compares partial sums of the Taylor series of the **finite
empirical sample** with that **same sample's** fully exponentially reweighted
mean. It measures how fast the sample's own series converges at
`lambda = 1`. It is not the uncertainty of the population cumulant
expansion, and it does not include sampling or overlap error. Those enter
through the standard errors of C_k and through the comparison with the direct
target mean (`error_vs_direct_target_*` columns).

## Interpretation

**C1 is the only clearly resolved correction in most cases.** For NNP to mW,
|C1|/SE is 5.5-9.4 at N = 64-512 and 3.4 at N=2048 (2.0 at N=1024). For
mW to NNP, C1 never reaches 3 SE (at most 2.1).

**C2-C4 are generally not statistically resolved with the available
frames.** No C2, C3, or C4 reaches 3 SE in any of the 12 groups; the largest
is 2.6 (C2 at N=64). The sign pattern and magnitude ordering of C2-C4 are
therefore not interpretable.

**Their uncertainty grows rapidly with N.** V = -beta*delta_U is extensive,
and Var(V) grows from 0.15 at N=64 to 5.0-5.2 at N=2048. The standard error of
C_k scales roughly as Var(V)^(k/2)/sqrt(M), so higher orders become noisy
fastest. For NNP to mW, SE(C4) grows from 1.5e-7 to 9.6e-5 eV/particle
(about 640x) between N=64 and N=2048, while SE(C1) grows only from 1.2e-5 to
4.3e-5. The underlying coefficients are not expected to grow with N,
because C_k combines extensive cumulants with the per-particle G. C1 is
indeed of similar size at all N.

**The overlap problem is not removed by the cumulant expansion.** It
reappears as a rapidly growing statistical cost for estimating the
higher-order coefficients.

**Convergence of the empirical series at lambda = 1.** For N <= 1024 the
sample's series approaches its full reweighted value: by order 12 the
residual is below 6e-8 eV/particle in both directions, and below 5e-10 for
N <= 512. At N=2048 the expansion is not practically convergent at
lambda = 1 over the accessible orders. For NNP to mW the coefficients do not
decrease through order 12 (C5 = -1.8e-4; C9-C12 are between 2.8e-5 and 4.8e-5), and
the residual stays between 3.9e-6 and 1.1e-4 for orders 4-12 (5.4e-5 at
order 12). For mW to NNP the residual is 9.6e-5 at order 4 and only falls
to 1.3e-6 by order 12. The fourth-order residuals at N=2048 (9.2e-5 and
9.6e-5) therefore do not measure a truncation error that a few more orders
would remove.
