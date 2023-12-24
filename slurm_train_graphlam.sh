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

PREPROCESS=false

# Load necessary modules
conda activate neural-ddp

if $PREPROCESS; then
    srun -ul python tools/create_static_features.py --boundaries 60
    srun -ul python tools/create_mesh.py --dataset "cosmo"
    srun -ul python tools/create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul python tools/create_parameter_weights.py --dataset "cosmo" --batch_size 12 --n_workers 8 --step_length 1
fi
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --val_interval 20 --epochs 40 --n_workers 8 --batch_size 12
    # --load saved_models/graph_lam-4x64-11_15_22_38_47/last.ckpt --resume_run '3gio4mcv'
