#!/bin/bash -l
#SBATCH --job-name=NeurWPredict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_pred_out.log
#SBATCH --error=lightning_logs/neurwp_pred_err.log
#SBATCH --time=03:00:00
#SBATCH --no-requeue


export PREPROCESS=true
export NORMALIZE=false

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    echo "Creating mesh"
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo" --plot 1
    echo "Creating grid features"
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        echo "Creating normalization weights"
        srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 32 --n_workers 8 --step_length 1
    fi
fi
# Load necessary modules
conda activate neural-lam


ulimit -c 0
export OMP_NUM_THREADS=16

srun -ul python train_model.py --load "wandb/example.ckpt" --dataset "cosmo" \
    --eval="predict" --subset_ds 1 --n_workers 2 --batch_size 6 --model "graph_lam"
