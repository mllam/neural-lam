#!/bin/bash -l
#SBATCH --job-name=Neural-LAM
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres:gpu=4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurallam_out_%j.log
#SBATCH --error=lightning_logs/neurallam_err_%j.log

# Load necessary modules or activate environment, for example:
conda activate neural-lam

srun -ul python train_model.py --val_interval 5 --epochs 20 --n_workers 8 --batch_size 12 --model hi_lam --graph hierarchical
