#!/bin/bash -l
#SBATCH --job-name=NLZarr
#SBATCH --nodes=1
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --output=lightning_logs/zarr_out.log
#SBATCH --error=lightning_logs/zarr_err.log
#SBATCH --mem=410G
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue

# Load necessary modules
conda activate neural-lam
ulimit -c 0
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul python create_zarr_archive.py
