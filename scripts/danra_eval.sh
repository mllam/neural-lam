#!/bin/bash

python -m neural_lam.train_model\
    --num_workers 2 \
    --precision bf16-mixed\
    --batch_size 1\
    --hidden_dim 300\
    --hidden_dim_grid 150\
    --time_delta_enc_dim 32\
    --config_path danra_model_config_ifs.yaml\
    --model hi_lam\
    --processor_layers 2\
    --graph_name rect_hi4\
    --num_nodes 1\
    --epochs 1\
    --ar_steps_eval 40\
    --val_steps_to_log 1 2 4 8 12 20 30 40\
    --load danra_model.ckpt\
    --eval test\
    --plot_vars pres_seasurface t2m u10m lwavr0m z700 r700
