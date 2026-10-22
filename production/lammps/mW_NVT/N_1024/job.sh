#!/bin/bash
#SBATCH --constraint=asc-vanilla
#SBATCH -N 1
#SBATCH -J LAMMPS_N1024
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=zen4_0768
#SBATCH --qos=zen4_0768
##SBATCH --account=p201232
#SBATCH --time=11:00:00
#SBATCH --output=out.%j
#SBATCH --error=err.%j

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lammps-build

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

mpirun -np 64 /home/sr41880/lammps-n2p2/build/lmp -in md.lmp
