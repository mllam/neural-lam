# Standard library
import os
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from neural_lam import utils
from neural_lam.models.base_graph_model import BaseGraphModel
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "base_graph": BaseGraphModel,
    "hi_lam_parallel": HiLAMParallel,
}


def main():
    # pylint: disable=too-many-branches
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )

    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate/predict"
        "(default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--resume_run", type=str, help="Run ID to resume (default: None)"
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        type=int,
        default=0,
        help="Train only on control member of ensemble data "
        "(default: 0 (False))",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test/predict) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    # Get args
    args = parser.parse_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
        "predict",
    ), f"Unknown eval setting: {args.eval}"

    # Set seed
    seed.seed_everything(args.seed)
    # Create datamodule
    data_module = WeatherDataModule(
        args.dataset,
        subset=args.subset_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)

    result = utils.init_wandb(args)

    if result is not None:
        logger = result
        checkpoint_dir = logger.experiment.dir
    else:
        logger = None
        checkpoint_dir = "lightning_logs"

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        verbose=True,
    )
    if args.eval:
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True
    utils.rank_zero_print("Arguments:")
    for arg in vars(args):
        utils.rank_zero_print(f"{arg}: {getattr(args, arg)}")

    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = int(
            os.environ.get("SLURM_GPUS_PER_NODE", torch.cuda.device_count())
        )
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    else:
        accelerator = "cpu"
        devices = 1
        num_nodes = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=(
            [checkpoint_callback] if checkpoint_callback is not None else []
        ),
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        use_distributed_sampler=use_distributed_sampler,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        profiler="simple",
        deterministic=True,
        limit_predict_batches=1,
        # num_sanity_val_steps=0
        # strategy="ddp",
        # limit_val_batches=0
        # fast_dev_run=True
    )
    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    # Check if the mode is evaluation (either 'val' or 'test')
    if args.eval in ["val", "test"]:
        data_module.split = args.eval
        trainer.test(model=model, datamodule=data_module, ckpt_path=args.load)

    # Check if the mode is prediction
    elif args.eval == "predict":
        data_module.split = "predict"
        trainer.predict(
            model=model,
            datamodule=data_module,
            return_predictions=True,
            ckpt_path=args.load,
        )
    # Default mode is training
    else:
        data_module.split = "train"
        if args.load:
            trainer.fit(
                model=model, datamodule=data_module, ckpt_path=args.load
            )
        else:
            trainer.fit(model=model, datamodule=data_module)

    # Print profiler
    print(trainer.profiler)  # pylint: disable=no-member


if __name__ == "__main__":
    main()
