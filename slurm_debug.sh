#!/bin/bash -l
#SBATCH --job-name=NeurWPd
#SBATCH --output=lightning_logs/neurwp_debug_out.log
#SBATCH --error=lightning_logs/neurwp_debug_err.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --mem=490G

export PREPROCESS=false

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=16

if [ "$PREPROCESS" = true ]; then
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo"
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 8 --n_workers 8 --step_length 1
fi

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --subset_ds 1 --n_workers 8 --batch_size 8 --model "graph_lam" \
    --epochs 1 --val_interval 1
