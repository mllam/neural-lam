#!/bin/bash

python -m neural_lam.train_model \
    --model hi_lam \
    --epochs 1 \
    --eval val \
    --n_example_pred 1 \
    --ar_steps_eval 120 \
    --val_steps_to_log 1 12 24 36 48 60 72 84 96 108 120 \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --num_nodes 1 \
    --batch_size 1 \
    --plot_vars "T_2M" "U_10M" \
    --precision bf16-mixed \
    --graph_name rectangular_hierarchical \
    --config_path cosmo_model_config_ifs.yaml \
    --load cosmo_model.ckpt
