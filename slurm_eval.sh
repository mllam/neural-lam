#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_eval_out.log
#SBATCH --error=lightning_logs/neurwp_eval_err.log
#SBATCH --time=03:00:00

conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --load "wandb/run-20231225_205644-qf82s2yb/files/latest-v1.ckpt" \
    --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 8 --batch_size 12
