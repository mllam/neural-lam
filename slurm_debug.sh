#!/bin/bash -l
#SBATCH --job-name=NeurWPd
#SBATCH --output=lightning_logs/neurwp_debug.out
#SBATCH --error=lightning_logs/neurwp_debug.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=a100-80gb
#SBATCH --account=s83

# Load necessary modules
conda activate neural-ddp

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --subset_ds 1 --n_workers 31 --batch_size 4 --model "graph_lam" \
    --epochs 6 --val_interval 2
