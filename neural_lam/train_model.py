# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# Local
from . import utils
from .datastore.mllam import MLLAMDatastore
from .datastore.multizarr import MultiZarrDatastore
from .datastore.npyfiles import NpyFilesDatastore
from .models.graph_lam import GraphLAM
from .models.hi_lam import HiLAM
from .models.hi_lam_parallel import HiLAMParallel
from .weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


def _init_datastore(datastore_kind, path):
    if datastore_kind == "multizarr":
        datastore = MultiZarrDatastore(root_path=path)
    elif datastore_kind == "npyfiles":
        datastore = NpyFilesDatastore(root_path=path)
    elif datastore_kind == "mllam":
        datastore = MLLAMDatastore(root_path=path)
    else:
        raise ValueError(f"Unknown datastore kind: {datastore_kind}")
    return datastore


def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--datastore-kind",
        type=str,
        choices=["multizarr", "npyfiles", "mllam"],
        default="multizarr",
        help="Kind of datastore to use (default: multizarr)",
    )
    parser.add_argument(
        "--datastore-path",
        type=str,
        default="tests/datastore_configs/multizarr",
        help="The root path for the datastore",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
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
        "--ar_steps_train",
        type=int,
        default=3,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 3)",
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
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 10)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )

    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural_lam",
        help="Wandb project name (default: neural_lam)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10, 15, 19],
        help="Steps to log val loss for (default: 1 2 3 5 10 15 19)",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--forcing-window-size",
        type=int,
        default=3,
        help="Number of time steps to use as input for forcing data",
    )
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)
    # Create datastore
    datastore = _init_datastore(
        datastore_kind=args.datastore_kind, path=args.datastore_path
    )
    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        standardize=True,
        forcing_window_size=args.forcing_window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    ModelClass = MODELS[args.model]
    model = ModelClass(
        args, datastore=datastore, forcing_window_size=args.forcing_window_size
    )

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.WandbLogger(
        project=args.wandb_project, name=run_name, config=args
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        devices=1,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, val_steps=args.val_steps_to_log
        )  # Do after wandb.init
        # TODO: should we save the datastore config here?
        # wandb.save()
    if args.eval:
        trainer.test(model=model, datamodule=data_module, ckpt_path=args.load)
    else:
        trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
