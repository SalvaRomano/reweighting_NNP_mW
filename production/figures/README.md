# Figure organization

- `Production/`: central collection of the figures currently selected for the
  paper. These are copies; the category-specific originals remain in their
  respective `production/` subfolders.
- `Bias/`: finite-sample bias, scaled-bias, and bias-versus-size diagnostics.
  The selected paper figure is under `Bias/production/`.
- `Reweighting/`: ESS, weight-distribution, overlap, thermodynamic-trace, and
  energy-error figures. Selected paper figures are under
  `Reweighting/production/`.
- `Cumulant/`: cumulant coefficients, contributions, and truncation errors.
- `Autocorrelation/`: NVT and NpT energy-autocorrelation figures.
- `Structural/`: RDF, coordination-number, and positional-correlation figures.

Figure-generating scripts default to these category directories so rerunning
an analysis does not recreate loose files at the top level.
