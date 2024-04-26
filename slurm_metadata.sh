#!/bin/bash -l
#SBATCH --job-name=Metadata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=pp-short
#SBATCH --account=s83
#SBATCH --output=lightning_logs/metadata_out.log
#SBATCH --error=lightning_logs/metadata_err.log
#SBATCH --time=00:03:00
#SBATCH --no-requeue

# Load necessary modules
conda activate neural-lam


ulimit -c 0
export OMP_NUM_THREADS=16

srun -ul python grib_modifier.py
