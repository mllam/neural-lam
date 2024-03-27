#!/bin/bash -l
#SBATCH --job-name=dask-job
#SBATCH --output=lightning_logs/%x_%j.out
#SBATCH --error=lightning_logs/%x_%j.err
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=444G
#SBATCH --time=5-00:00:00
#SBATCH --exclusive

# Activate Python environment
conda activate neural-lam

# Set up environment variables
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=100000s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=100000s

srun -ul python create_zarr_archive.py
