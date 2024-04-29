#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --account=s83
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --partition=postproc
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurwp_param_out.log
#SBATCH --error=lightning_logs/neurwp_param_err.log

export DATASET="cosmo"

# Load necessary modules
conda activate neural-lam

ntasks=72
batch_size=1
subset=$((365 * 24))
next_subset=$(((subset + ntasks * batch_size - 1) / (ntasks * batch_size) * (ntasks * batch_size)))
echo "Creating normalization weights base on $next_subset timesteps..."

srun -ul --ntasks $ntasks python create_parameter_weights.py --dataset $DATASET --batch_size $batch_size --n_workers 1 --subset $next_subset
