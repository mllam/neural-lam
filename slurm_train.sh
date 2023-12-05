#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp.out
#SBATCH --error=lightning_logs/neurwp.err
#SBATCH --mem=490G

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    # --load saved_models/graph_lam-4x64-11_15_22_38_47/last.ckpt \
    # --resume_run '3gio4mcv' \
    --dataset "cosmo" --val_interval 10 --epochs 100 --n_workers 8 \
    --batch_size 8 --model "graph_lam"