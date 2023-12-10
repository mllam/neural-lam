#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_eval_out.log
#SBATCH --error=lightning_logs/neurwp_eval_err.log

conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=24

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --load "saved_models/graph_lam-4x64-12_08_18_59_10/latest-v1.ckpt" \
    --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 12 --batch_size 1
