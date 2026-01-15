# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastores
from .models import GraphLAM, HiLAM, HiLAMParallel
from .weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


@logger.catch
def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
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
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP (default: 1)",
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
        action="store_true",
        help="If full training state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps to run before starting"
        " training (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model (default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim_grid",
        type=int,
        help=(
            "Dimensionality of hidden representations related to grid nodes "
            "(grid encodings and in grid-level MLPs)"
            "(default: None, use same as hidden_dim)"
        ),
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
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--shared_grid_embedder",
        action="store_true",  # Default to separate embedders
        help="If the same embedder MLP should be used for interior and boundary"
        " grid nodes. Note that this requires the same dimensionality for "
        "both kinds of grid inputs. (default: False (no))",
    )
    parser.add_argument(
        "--time_delta_enc_dim",
        type=int,
        help="Dimensionality of positional encoding for time deltas of boundary"
        " forcing. If None, same as hidden_dim. If given, must be even "
        "(default: None)",
    )
    parser.add_argument(
        "--dynamic_time_deltas",
        action="store_true",
        help="If models should use dynamically computed time-deltas between"
        "interior and boundary time steps (default: False (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 1)",
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
        "--min_lr",
        type=float,
        default=1e-4,
        help="Minimum learning rate for cosine annealing (default: 1e-4)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="If gradient checkpointing should be used in-between each "
        "unrolling step (default: false)",
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
        default=11,
        help="Number of steps to unroll prediction for during evaluation "
        "(default: 10)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )
    parser.add_argument(
        "--eval_init_times",
        nargs="*",
        default=[0, 12],
        help="List of init times (UTC) where validation and evaluation "
        "forecasts should be started from (default: 0, 12)",
    )
    parser.add_argument(
        "--save_eval_to_zarr_path",
        type=str,
        help="Save evaluation results to zarr dataset at given path ",
    )
    parser.add_argument(
        "--save_eval_to_pt_path",
        type=str,
        help="Save results of 'n_example_pred' to pt dataset at given path ",
    )
    parser.add_argument(
        "--plot_vars",
        nargs="+",
        type=str,
        default=["Surface elevation", "U velocity", "V velocity"],
        help="""List of variables to plot during eval
        (default: Surface elevation, U velocity, V velocity)""",
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
        default=[1, 2, 3, 5, 10, 11],
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
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as forcing input (default: 1)",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as forcing input (default: 1)",
    )
    parser.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as boundary input (default: 1)",
    )
    parser.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as boundary input "
        "(default: 1)",
    )
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    # Asserts for arguments
    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"
    for step in args.val_steps_to_log:
        assert step <= args.ar_steps_eval, (
            f"Can not log validation step {step} when validation is "
            f"only unrolled {args.ar_steps_eval} steps."
        )
    assert (
        args.load or not args.restore_opt
    ), "Can not restore opt state when not loading a checkpoint"

    # Infer Run mode
    eval_only = args.eval and args.load and not args.restore_opt
    train_and_eval = args.eval and not eval_only

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore to use
    config, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        num_past_boundary_steps=args.num_past_boundary_steps,
        num_future_boundary_steps=args.num_future_boundary_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Make sure that dataset provided for eval contains correct split
        eval_split=args.eval if args.eval is not None else "test",
        eval_init_times=args.eval_init_times,
        dynamic_time_deltas=args.dynamic_time_deltas,
        excluded_intervals=config.training.excluded_intervals,
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
    if args.load and not args.restore_opt:
        # Restore only model weights, not opt setup
        model = ModelClass.load_from_checkpoint(
            args.load,
            args=args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )
    else:
        model = ModelClass(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )

    if eval_only:
        run_mode = f"eval-{args.eval}"
    elif train_and_eval:
        run_mode = f"train+eval-{args.eval}"
    else:
        run_mode = "train"

    run_name = (
        f"{run_mode}-"
        f"{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    checkpoint_root = "saved_models"
    if eval_only:
        checkpoint_dir = f"{checkpoint_root}/eval_only/{run_name}"
    else:
        checkpoint_dir = f"{checkpoint_root}/{run_name}"

    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="min_val_mean_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )

    # Checkpoint for min val loss at step ar_steps_train
    possible_monitor_steps = [
        step for step in args.val_steps_to_log if step <= args.ar_steps_train
    ]
    assert possible_monitor_steps, (
        "Can not save checkpoints as no validation loss is logged for "
        f"step {args.ar_steps_train} or earlier."
    )
    # Choose step closest to ar_steps_train
    monitored_unroll_step = max(possible_monitor_steps)
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"min_val_loss_unroll{monitored_unroll_step}",
            monitor=f"val_loss_unroll{monitored_unroll_step}",
            mode="min",
            save_last=False,  # Only need one save_last=True
        )
    )

    logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=dict(training=vars(args), datastore=datastore._config),
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        num_nodes=args.num_nodes,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_steps,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, val_steps=args.val_steps_to_log
        )  # Do after wandb.init

    eval_only = (
        args.eval is not None and args.load is not None and not args.restore_opt
    )

    # Resume training if restore_opt is set
    ckpt_for_fit = args.load if args.restore_opt else None

    if not eval_only:
        # 1) Train (fresh or resumed)
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_for_fit,
        )

    # 2) Evaluate
    if args.eval:
        # Evaluation checkpoint logic
        if eval_only:
            ckpt_path = args.load
        else:
            ckpt_path = "best"

        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    main()
