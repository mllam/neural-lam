#!/bin/bash
#SBATCH --job-name=NeurWP
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp.out
#SBATCH --error=lightning_logs/neurwp.err
#SBATCH --exclusive
#SBATCH --mem=490G

# Load necessary modules
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --subset_ds 0 --val_interval 100 --epochs 200 --n_workers 8 \
    --batch_size 8 --model "graph_lam" \
    --load "saved_models/graph_lam-4x64-11_02_22_55_35/min_val_loss.ckpt"
