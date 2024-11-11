python train_model.py\
    --dataset meps\
    --model graph_efm\
    --n_example_pred 0\
    --graph multiscale\
    --n_workers 16\
    --hidden_dim 128\
    --processor_layers 4\
    --prior_processor_layers 2\
    --encoder_processor_layers 2\
    --output_std 1\
    --ensemble_size 100\
    --batch_size 1\
    --load paper_checkpoints/graph_efm_ms.ckpt\
    --eval test\

