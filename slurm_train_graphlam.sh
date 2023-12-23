#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --mem=490G
#SBATCH --exclusive

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=24

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --val_interval 20 --epochs 40 --n_workers 12 --batch_size 12
    # --load saved_models/graph_lam-4x64-11_15_22_38_47/last.ckpt --resume_run '3gio4mcv'
