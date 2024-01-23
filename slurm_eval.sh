#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_eval_out.log
#SBATCH --error=lightning_logs/neurwp_eval_err.log
#SBATCH --time=03:00:00

export PREPROCESS=true
export NORMALIZE=false

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo" --plot 1
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 32 --n_workers 8 --step_length 1
    fi
fi

ulimit -c 0
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul python train_model.py --load "wandb/example.ckpt" --dataset "cosmo" \
    --eval="test" --subset_ds 1 --n_workers 2 --batch_size 6 --wandb_mode "offline"
