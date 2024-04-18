#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --no-requeue
#SBATCH --output=lightning_logs/neurwp_eval_out.log
#SBATCH --error=lightning_logs/neurwp_eval_err.log

export PREPROCESS=true
export NORMALIZE=false
export DATASET="cosmo"

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    python create_static_features.py --boundaries 60
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

srun -ul python train_model.py --load "wandb/example.ckpt" --dataset $DATASET \
    --eval="test" --subset_ds 1 --n_workers 2 --batch_size 6
