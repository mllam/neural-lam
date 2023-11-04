#!/bin/bash
#SBATCH --job-name=NeurWPe
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=lightning_logs/neurwp_eval.out
#SBATCH --error=lightning_logs/neurwp_eval.err

# Load necessary modules
source ${SCRATCH}/miniforge3/etc/profile.d/conda.sh
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=4

NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

# Run the script with torchrun
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS train_model.py \
    --load "saved_models/graph_lam-4x64-11_02_22_55_35/min_val_loss.ckpt" \
    --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 31 --batch_size 4 --model "graph_lam"
