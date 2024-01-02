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

export PREPROCESS=false

# Load necessary modules
conda activate neural-ddp

if [ "$PREPROCESS" = true ]; then
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo"
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 12 --n_workers 8 --step_length 1
fi

ulimit -c 0
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --load "wandb/run-20231231_164801-xkwwdmb0/files/latest-v1.ckpt" \
    --dataset "cosmo" --eval="test" --subset_ds 1 --n_workers 4 --batch_size 6
