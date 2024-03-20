#!/bin/bash -l
#SBATCH --job-name=NeurWPredict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=pp-short
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_pred_out.log
#SBATCH --error=lightning_logs/neurwp_pred_err.log
#SBATCH --time=00:15:00
#SBATCH --no-requeue

export PREPROCESS=false
export NORMALIZE=false

# Load necessary modules
conda activate neural-lam


ulimit -c 0
export OMP_NUM_THREADS=16

srun -ul python train_model.py --load "wandb/example.ckpt" --dataset "cosmo" \
    --eval="predict" --subset_ds 1 --n_workers 2 --batch_size 6 --model "graph_lam"