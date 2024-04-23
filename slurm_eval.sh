#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=normal
#SBATCH --mem=375G
#SBATCH --no-requeue
#SBATCH --output=lightning_logs/neurwp_eval_out.log
#SBATCH --error=lightning_logs/neurwp_eval_err.log

export PREPROCESS=true
export NORMALIZE=false
export DATASET="cosmo"
export MODEL="hi_lam"

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    python create_static_features.py --boundaries 60 --dataset $DATASET
    if [ "$MODEL" = "hi_lam" ]; then
        echo "Creating hierarchical mesh"
        python create_mesh.py --dataset $DATASET --plot 1 --graph hierarchical --levels 4 --hierarchical 1
    else
        echo "Creating flat mesh"
        python create_mesh.py --dataset $DATASET --plot 1
    fi
    echo "Creating grid features"
    python create_grid_features.py --dataset $DATASET
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        echo "Creating normalization weights"
        python create_parameter_weights.py --dataset $DATASET --batch_size 32 --n_workers 8 --step_length 1
    fi
fi

echo "Evaluating model"
if [ "$MODEL" = "hi_lam" ]; then
    srun -ul python train_model.py --dataset $DATASET --val_interval 2 --epochs 1 --n_workers 4 --batch_size 1 --subset_ds 1 --model hi_lam --graph hierarchical --load wandb/example.ckpt --eval="test"
else
    srun -ul python train_model.py --dataset $DATASET --val_interval 2 --epochs 1 --n_workers 4 --batch_size 1 --subset_ds 1 --load "wandb/example.ckpt" --eval="test"
fi
