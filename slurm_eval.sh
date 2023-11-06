#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_eval.out
#SBATCH --error=lightning_logs/neurwp_eval.err

# Load necessary modules
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --load "saved_models/hi_lam-4x64-11_04_00_58_26/min_val_loss.ckpt" \
    --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 8 --batch_size 8 \
    --model "hi_lam" --graph hierarchical
