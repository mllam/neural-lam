#!/bin/bash -l
#SBATCH --job-name=NeurWPd
#SBATCH --output=lightning_logs/neurwp_debug_out.log
#SBATCH --error=lightning_logs/neurwp_debug_err.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --mem=490G

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=128

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --subset_ds 1 --n_workers 128 --batch_size 4 --model "graph_lam" \
    --epochs 1 --val_interval 1
