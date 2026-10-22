#!/bin/bash
set -euo pipefail
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_64"
"$1" -in rerun.lmp -screen screen.out
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_128"
"$1" -in rerun.lmp -screen screen.out
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_256"
"$1" -in rerun.lmp -screen screen.out
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_512"
"$1" -in rerun.lmp -screen screen.out
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_1024"
"$1" -in rerun.lmp -screen screen.out
cd "/data/home/salvatorer26/projects/reweighting/production/results/full_mw_trajectory/N_2048"
"$1" -in rerun.lmp -screen screen.out
