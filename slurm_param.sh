#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --account=s83
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --partition=postproc
#SBATCH --mem=375G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurwp_param_out.log
#SBATCH --error=lightning_logs/neurwp_param_err.log

export DATASET="cosmo"

# Load necessary modules
conda activate neural-lam

echo "Creating normalization weights"
srun -ul --ntasks 32 python create_parameter_weights.py --dataset $DATASET --batch_size 4 --n_workers 4 --subset 8760
