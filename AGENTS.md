# Repository Guidelines
This is a scientific data-analysis repository.

## Project Structure & Module Organization

This repository contains water simulation, dataset preparation, and neural-network-potential training artifacts. `simulation/` holds LAMMPS inputs, parsers, and density/negative-pressure run folders. `dataset/` contains prepared `input.data` collections and pruning scripts. `train_NNP/` and `train_NNP_final/` contain n2p2 training inputs, committee run folders, analysis scripts, plots, and notebooks. `nnp-water/` and `new_nnp/` store trained model inputs, scaling data, and weights. `trial/` is for exploratory LAMMPS and NN runs.

Most Python and shell scripts use relative paths and should be run from the directory where they live unless the script states otherwise.

## Build, Test, and Development Commands

There is no package build system at the repository root. Use scripts directly:

- `python train_NNP/plot_learning_curves.py`: generate learning-curve plots from `run_*/train/learning-curve.out`.
- `python train_NNP/compare_energy_forces.py`: create parity plots from `run_*/train/dataset/{energy,forces}.comp`.
- `python simulation/mW_dataset_negative_pressure/parser_lammps.py`: convert LAMMPS trajectory and energy outputs when configured in the script.
- `bash train_NNP/starter.sh`: create `run_1` through `run_4` training folders from local `input.nn`, `input.data`, and `validation.data`.

Run n2p2 and LAMMPS commands from the relevant run directory so generated logs remain local.

## Coding Style & Naming Conventions

Python scripts are simple procedural tools. Use 4-space indentation, `snake_case` names, and explicit top-level user settings for paths and plotting options. Keep generated outputs in run-specific folders such as `run_1/`, `dataset/`, `comparison_plots/`, or `learning_curves/`. Shell scripts should start with `#!/bin/bash`, keep parameters near the top, and avoid hard-coded seeds unless documented.

## Testing Guidelines

No automated test framework is configured. For Python changes, run the edited script on a small existing dataset and confirm expected files are created. For parsers, verify frame counts and energy counts match before committing generated `input.data`. For plotting scripts, check that output PNGs update in the intended directory only.

## Commit & Pull Request Guidelines

This branch has no commit history, so use clear imperative commit messages such as `Add LAMMPS parser validation` or `Update NNP comparison plots`. Pull requests should describe the scientific workflow affected, list commands run, note any new large generated files, and include representative plot screenshots when analysis output changes.

## Data & Configuration Tips

Avoid overwriting trained weights, `scaling.data`, LAMMPS restarts, or large trajectory files unless the change is intentional. Prefer adding new run directories over mutating prior results, and document random seeds used for new training committees.


## Rules

- Do not modify raw data files.
- Put reusable code in `src/`.
- Put scripts in `scripts/`.
- Put figures in `figures/`.
- Put processed outputs in `results/`.
- Prefer simple, readable Python over clever code.
- Use numpy, pandas, scipy, matplotlib.
- Always add comments for nontrivial analysis steps.
- Do not hard-code paths if avoidable.
- Before changing files, explain the plan briefly.
- After changes, run the script if possible and show the command used.


