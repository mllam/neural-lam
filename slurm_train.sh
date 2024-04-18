#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=normal
#SBATCH --mem=375G
#SBATCH --no-requeue
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log

export PREPROCESS=false
export NORMALIZE=false
export DATASET="cosmo"

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    python create_static_features.py --boundaries 60 --dataset "cosmo"
    echo "Creating mesh"
    python create_mesh.py --dataset $DATASET --plot 1
    echo "Creating grid features"
    python create_grid_features.py --dataset $DATASET
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        echo "Creating normalization weights"
        python create_parameter_weights.py --dataset $DATASET --batch_size 32 --n_workers 8 --step_length 1
    fi
fi

ulimit -c 0
export OMP_NUM_THREADS=16

srun -ul python train_model.py --dataset $DATASET --val_interval 5 --epochs 10 --n_workers 1 --batch_size 1 --subset_ds 1
