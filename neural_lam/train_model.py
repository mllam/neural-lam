"""CLI entry point for training Neural-LAM models."""

# Standard library
import json
import os
import random
import shutil
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
# for logging the model:
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastore
from .gnn_layers import GNN_TYPES
from .models import MODELS, ARForecaster, ForecasterModule
from .weather_dataset import WeatherDataModule


class AdaptiveHelpFormatter(ArgumentDefaultsHelpFormatter):
    """``--help`` formatter that scales the column width to the terminal."""

    def __init__(self, prog):
        """Pick a help-column width based on the current terminal size."""
        terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        width = max(80, min(terminal_width, 120))
        help_position = min(44, width // 3)
        super().__init__(
            prog,
            max_help_position=help_position,
            width=width,
        )


def load_forecaster_module_from_checkpoint(ckpt_path, config, datastore):
    """
    Reconstruct a ForecasterModule from a checkpoint without requiring the
    caller to know the original architecture kwargs.

    The checkpoint must have been saved with args in hyper_parameters (i.e.
    created via train_model.main), so that model class and architecture kwargs
    can be recovered automatically.
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    args = ckpt["hyper_parameters"]["args"]
    predictor_class = MODELS[args.model]
    predictor = predictor_class(
        datastore=datastore,
        graph_name=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        processor_layers=args.processor_layers,
        mesh_aggr=args.mesh_aggr,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        output_std=args.output_std,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )
    forecaster = ARForecaster(predictor, datastore)
    return ForecasterModule.load_from_checkpoint(
        ckpt_path,
        forecaster=forecaster,
        datastore=datastore,
        weights_only=False,
    )


@logger.catch
def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(
        description="Train or evaluate MLWP models for LAM",
        formatter_class=AdaptiveHelpFormatter,
    )

    # Core Configuration
    core_group = parser.add_argument_group("Core Configuration")
    core_group.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
        required=True,
    )
    core_group.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate",
        choices=MODELS.keys(),
    )
    core_group.add_argument("--seed", type=int, default=42, help="random seed")

    # Runtime & Device Settings
    runtime_group = parser.add_argument_group("Runtime & Device Settings")
    runtime_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader",
    )
    runtime_group.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP",
    )
    runtime_group.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training. Can be the string 'auto' or a list "
        "of integer id's corresponding to the desired devices, e.g. "
        "'--devices 0 1'. Note that this cannot be used with SLURM, instead "
        "set 'ntasks-per-node' in the slurm setup",
    )
    runtime_group.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16)",
    )
    runtime_group.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from",
    )
    runtime_group.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model",
    )

    # Model architecture
    arch_group = parser.add_argument_group("Model Architecture")
    arch_group.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model",
    )
    arch_group.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations",
    )
    arch_group.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs",
    )
    arch_group.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN",
    )
    arch_group.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean)",
    )
    arch_group.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions",
    )
    arch_group.add_argument(
        "--g2m_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for grid-to-mesh encoding",
    )
    arch_group.add_argument(
        "--m2g_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for mesh-to-grid decoding",
    )
    arch_group.add_argument(
        "--mesh_up_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for upward mesh message passing in hierarchical models",
    )
    arch_group.add_argument(
        "--mesh_down_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for downward mesh message passing in "
        "hierarchical models",
    )

    # Training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=4, help="batch size"
    )

    train_group.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function",
    )
    train_group.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py",
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate"
    )
    train_group.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run",
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test). If not given, "
        "train model.",
        choices=["val", "test"],
    )
    eval_group.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for during evaluation",
    )
    eval_group.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation",
    )
    eval_group.add_argument(
        "--create_gif",
        action="store_true",
        help="If set, create GIF animations from prediction PNG frames and "
        "save to disk. PNGs are always created and logged to wandb/mlflow.",
    )

    # Logger Settings
    logger_group = parser.add_argument_group("Logger Settings")
    logger_group.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logger to use for training (wandb/mlflow)",
    )
    logger_group.add_argument(
        "--logger-project",
        type=str,
        default="neural_lam",
        help="Logger project name, for eg. Wandb",
    )
    logger_group.add_argument(
        "--logger_run_name",
        type=str,
        default=None,
        help="""Logger run name, for e.g. MLFlow (with default value `None`
          neural-lam's default format string is used)""",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs",
        help="Root directory under which per-run output dirs (checkpoints, "
        "logger files, plots) are written as `<runs_root>/<run_name>/`",
    )

    logger_group.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Wandb run ID to use. If the run ID already exists in the "
        "project, W&B resumes that run. If it does not exist, W&B creates "
        "a new run with that ID. Useful on HPC systems with limited job "
        "runtimes or that may crash, allowing training to be continued "
        "across multiple job submissions.",
    )

    # Metrics & Monitoring (logger-agnostic: applies to both wandb and mlflow)
    metrics_group = parser.add_argument_group("Metrics & Monitoring")
    metrics_group.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
        help="Steps to log val loss for",
    )
    metrics_group.add_argument(
        "--train_steps_to_log",
        nargs="+",
        type=int,
        default=[],
        help="Steps to log train loss for during training (optional)",
    )
    metrics_group.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    metrics_group.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )

    # Data Loading & Forcing
    data_group = parser.add_argument_group("Data Loading & Forcing")
    data_group.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as input for forcing data",
    )
    data_group.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as input for forcing data",
    )
    data_group.add_argument(
        "--load_single_member",
        action="store_true",
        help=(
            "If set, only use ensemble member 0 instead of treating all "
            "ensemble members as independent samples."
        ),
    )
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    # Check that config only specifies logging for lead times that exist
    # Check --val_steps_to_log
    for step in args.val_steps_to_log:
        if step > args.ar_steps_eval:
            raise ValueError(
                f"Can not log validation step {step} when validation is "
                f"only unrolled {args.ar_steps_eval} steps. Adjust "
                "--val_steps_to_log."
            )
    # Check --train_steps_to_log
    for step in args.train_steps_to_log:
        if step > args.ar_steps_train:
            raise ValueError(
                f"Can not log training step {step} when training is "
                f"only unrolled {args.ar_steps_train} steps. Adjust "
                "--train_steps_to_log."
            )
    # Check --var_leads_metric_watch
    for var_i, leads in args.var_leads_metrics_watch.items():
        for step in leads:
            if step > args.ar_steps_eval:
                raise ValueError(
                    f"Can not log validation step {step} for variable "
                    f"{var_i} when validation is only unrolled "
                    f"{args.ar_steps_eval} steps. Adjust "
                    "--var_leads_metric_watch."
                )

    if args.eval and not args.load:
        logger.warning(
            "Evaluation (--eval) without --load: no checkpoint will be loaded.",
        )

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore to use
    config, datastore = load_config_and_datastore(config_path=args.config_path)

    # Check --var_leads_metrics_watch variable indices against the datastore
    # so users get an immediate error instead of an IndexError deep in the
    # first validation epoch.
    state_var_names = datastore.get_vars_names(category="state")
    for var_i in args.var_leads_metrics_watch:
        if not 0 <= var_i < len(state_var_names):
            raise ValueError(
                f"Invalid state variable index {var_i} in "
                f"--var_leads_metrics_watch. Index must be between 0 and "
                f"{len(state_var_names) - 1} (datastore has "
                f"{len(state_var_names)} state variables)."
            )

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        load_single_member=args.load_single_member,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_split=args.eval or "test",
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Set devices to use
    if args.devices == ["auto"]:
        devices = "auto"
    else:
        try:
            devices = [int(i) for i in args.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    predictor_class = MODELS[args.model]
    predictor = predictor_class(
        datastore=datastore,
        graph_name=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        processor_layers=args.processor_layers,
        mesh_aggr=args.mesh_aggr,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        output_std=args.output_std,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
        g2m_gnn_type=args.g2m_gnn_type,
        m2g_gnn_type=args.m2g_gnn_type,
        mesh_up_gnn_type=args.mesh_up_gnn_type,
        mesh_down_gnn_type=args.mesh_down_gnn_type,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss=args.loss,
        lr=args.lr,
        restore_opt=args.restore_opt,
        n_example_pred=args.n_example_pred,
        create_gif=args.create_gif,
        val_steps_to_log=args.val_steps_to_log,
        train_steps_to_log=args.train_steps_to_log,
        metrics_watch=args.metrics_watch,
        var_leads_metrics_watch=args.var_leads_metrics_watch,
        args=args,
    )

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"

    if args.logger_run_name:
        run_name = args.logger_run_name
    else:
        run_name = (
            f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
            f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
        )

    run_dir = os.path.join(args.runs_root, run_name)

    training_logger = utils.setup_training_logger(
        datastore=datastore, args=args, run_name=run_name, run_dir=run_dir
    )

    # Two separate callbacks decouple "best validated model" from
    # "rescue / resume" snapshots: long HPC jobs that crash or time out
    # between validations can still resume from a recent train-epoch
    # checkpoint instead of losing all progress since the last validation.
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    latest_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="last",
        monitor=None,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        default_root_dir=run_dir,
        strategy="auto",
        accelerator=device_name,
        num_nodes=args.num_nodes,
        devices=devices,
        logger=training_logger,
        log_every_n_steps=1,
        callbacks=[val_checkpoint, latest_checkpoint],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_training_logger_metrics(
            training_logger, val_steps=args.val_steps_to_log
        )  # Do after initializing logger
    if args.eval:
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=args.load,
        )
    else:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=args.load)


if __name__ == "__main__":
    main()
