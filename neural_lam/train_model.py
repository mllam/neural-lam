"""CLI entry point for training Neural-LAM models."""

# Standard library
import json
import os
import random
import shutil
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from typing import Any, cast

# Third-party
# for logging the model:
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import (
    ComputeConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    NeuralLAMConfig,
    TrainRunConfig,
    load_config_and_datastore,
)
from .datastore.base import BaseDatastore, BaseRegularGridDatastore
from .gnn_layers import GNN_TYPES
from .models import (
    MODELS,
    ARForecaster,
    BaseHiGraphModel,
    ForecasterModule,
)
from .weather_dataset import WeatherDataModule


def build_predictor(
    predictor_class: type,
    model_config: Any,
    config: NeuralLAMConfig,
    datastore: BaseDatastore,
    num_past_forcing_steps: int = 1,
    num_future_forcing_steps: int = 1,
) -> Any:
    """
    Instantiate a step predictor with the GNN kwargs its family accepts.

    Hierarchical GNN kwargs are only passed to ``BaseHiGraphModel``
    subclasses, gating on the class hierarchy so that future hierarchical
    models are covered without maintaining a model-name list. GNN type
    arguments fall back to ``InteractionNet`` for checkpoints saved before
    those CLI flags existed.
    """
    past_forcing = getattr(
        model_config, "num_past_forcing_steps", num_past_forcing_steps
    )
    future_forcing = getattr(
        model_config, "num_future_forcing_steps", num_future_forcing_steps
    )
    kwargs = dict(
        datastore=datastore,
        graph_name=getattr(model_config, "graph", "multiscale"),
        hidden_dim=getattr(model_config, "hidden_dim", 64),
        hidden_layers=getattr(model_config, "hidden_layers", 1),
        processor_layers=getattr(model_config, "processor_layers", 4),
        mesh_aggr=getattr(model_config, "mesh_aggr", "sum"),
        num_past_forcing_steps=past_forcing,
        num_future_forcing_steps=future_forcing,
        output_std=getattr(model_config, "output_std", False),
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
        g2m_gnn_type=getattr(model_config, "g2m_gnn_type", "InteractionNet"),
        m2g_gnn_type=getattr(model_config, "m2g_gnn_type", "InteractionNet"),
    )
    if issubclass(predictor_class, BaseHiGraphModel):
        kwargs["mesh_up_gnn_type"] = getattr(
            model_config, "mesh_up_gnn_type", "InteractionNet"
        )
        kwargs["mesh_down_gnn_type"] = getattr(
            model_config, "mesh_down_gnn_type", "InteractionNet"
        )
    return predictor_class(**kwargs)


class AdaptiveHelpFormatter(ArgumentDefaultsHelpFormatter):
    """``--help`` formatter that scales the column width to the terminal."""

    def __init__(self, prog: str) -> None:
        """Pick a help-column width based on the current terminal size."""
        terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        width = max(80, min(terminal_width, 120))
        help_position = min(44, width // 3)
        super().__init__(
            prog,
            max_help_position=help_position,
            width=width,
        )


def load_forecaster_module_from_checkpoint(
    ckpt_path: str,
    config: NeuralLAMConfig,
    datastore: BaseDatastore,
) -> ForecasterModule:
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
    predictor = build_predictor(predictor_class, args, config, datastore)
    forecaster = ARForecaster(predictor, datastore)
    return ForecasterModule.load_from_checkpoint(
        ckpt_path,
        forecaster=forecaster,
        datastore=datastore,
        weights_only=False,
    )


def build_parser() -> ArgumentParser:
    """Build the argument parser for training and evaluating models."""

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
        help="GNN type for grid-to-mesh encoding. Applies to all models, "
        "including the probabilistic Graph-EFM model",
    )
    arch_group.add_argument(
        "--m2g_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for mesh-to-grid decoding. Applies to all models, "
        "including the probabilistic Graph-EFM model",
    )
    arch_group.add_argument(
        "--mesh_up_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for upward mesh message passing in hierarchical "
        "models. Only affects Hi-LAM; the probabilistic Graph-EFM model "
        "hard-codes its mesh-up GNN types",
    )
    arch_group.add_argument(
        "--mesh_down_gnn_type",
        type=str,
        default="InteractionNet",
        choices=list(GNN_TYPES.keys()),
        help="GNN type for downward mesh message passing in hierarchical "
        "models. Only affects Hi-LAM; the probabilistic Graph-EFM model "
        "hard-codes its mesh-down GNN type",
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

    train_group.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=2,
        help="Number of sanity validation steps to run before training",
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
    return parser


def fit(
    model_config: ModelConfig,
    train_config: TrainRunConfig,
    data_config: DataConfig,
    compute_config: ComputeConfig,
    logging_config: LoggingConfig,
    *,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    config_path: str | None = None,
    args: Any | None = None,
) -> Any:
    """
    Run training or evaluation using strongly-typed configuration dataclasses.

    Parameters
    ----------
    model_config : ModelConfig
        Model architecture and graph configuration.
    train_config : TrainRunConfig
        Training hyperparameters and evaluation settings.
    data_config : DataConfig
        Data loading and forcing parameters.
    compute_config : ComputeConfig
        Compute devices, seed, and precision settings.
    logging_config : LoggingConfig
        Experiment tracking and logging settings.
    config : NeuralLAMConfig or None, optional
        Loaded Neural-LAM configuration.
    datastore : BaseDatastore or None, optional
        Initialized datastore instance.
    config_path : str or None, optional
        Path to the configuration file, used if config or datastore is None.
    args : Any or None, optional
        Legacy argparse Namespace for checkpoint backward compatibility.

    Returns
    -------
    Run
        Object containing paths to run outputs and saved checkpoints.
    """
    for phase, max_steps in [
        ("train", train_config.ar_steps_train),
        ("val", train_config.ar_steps_eval),
    ]:
        steps = (
            train_config.train_steps_to_log
            if phase == "train"
            else train_config.val_steps_to_log
        )
        for step in steps:
            if not 1 <= step <= max_steps:
                raise ValueError(
                    f"Can not log {phase} step {step}: must be between 1 "
                    f"and {max_steps}, the number of unrolled steps during "
                    f"{phase} phase."
                )

    for var_i, leads in train_config.var_leads_metrics_watch.items():
        for step in leads:
            if not 1 <= step <= train_config.ar_steps_eval:
                raise ValueError(
                    f"Can not log validation step {step} for variable "
                    f"{var_i}: must be between 1 and "
                    f"{train_config.ar_steps_eval}, the number of unrolled "
                    "validation steps."
                )

    if train_config.eval and not train_config.load:
        logger.warning(
            "Evaluation without load checkpoint: no checkpoint will be loaded. "
            "Use --load <checkpoint> to load a checkpoint."
        )

    random_run_id = random.randint(0, 9999)
    seed.seed_everything(compute_config.seed, workers=True)

    if config is None or datastore is None:
        if config_path is None:
            raise ValueError(
                "Either (config and datastore) or config_path must be provided."
            )
        loaded_config, loaded_datastore = load_config_and_datastore(
            config_path=config_path
        )
        config = config or loaded_config
        datastore = datastore or loaded_datastore

    state_var_names = datastore.get_vars_names(category="state")
    for var_i in train_config.var_leads_metrics_watch:
        if not 0 <= var_i < len(state_var_names):
            raise ValueError(
                f"Invalid state variable index {var_i} in "
                f"var_leads_metrics_watch. Index must be between 0 and "
                f"{len(state_var_names) - 1} (datastore has "
                f"{len(state_var_names)} state variables)."
            )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=train_config.ar_steps_train,
        ar_steps_eval=train_config.ar_steps_eval,
        num_past_forcing_steps=data_config.num_past_forcing_steps,
        num_future_forcing_steps=data_config.num_future_forcing_steps,
        load_single_member=data_config.load_single_member,
        batch_size=train_config.batch_size,
        num_workers=data_config.num_workers,
        eval_split=train_config.eval or "test",
    )

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    devices: str | list[int]
    if compute_config.devices == "auto" or compute_config.devices == ["auto"]:
        devices = "auto"
    elif isinstance(compute_config.devices, list):
        try:
            devices = [int(i) for i in compute_config.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")
    else:
        devices = str(compute_config.devices)

    predictor_class = MODELS[model_config.model]
    predictor = build_predictor(
        predictor_class,
        model_config,
        config,
        datastore,
        data_config.num_past_forcing_steps,
        data_config.num_future_forcing_steps,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss=train_config.loss,
        lr=train_config.lr,
        restore_opt=train_config.restore_opt,
        n_example_pred=train_config.n_example_pred,
        create_gif=train_config.create_gif,
        val_steps_to_log=train_config.val_steps_to_log,
        train_steps_to_log=train_config.train_steps_to_log,
        metrics_watch=train_config.metrics_watch,
        var_leads_metrics_watch=train_config.var_leads_metrics_watch,
        args=args,
    )

    prefix = f"eval-{train_config.eval}-" if train_config.eval else "train-"
    if logging_config.logger_run_name:
        run_name = logging_config.logger_run_name
    else:
        run_name = (
            f"{prefix}{model_config.model}-"
            f"{model_config.processor_layers}x{model_config.hidden_dim}-"
            f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
        )

    run_dir = os.path.join(logging_config.runs_root, run_name)

    training_logger = utils.setup_training_logger(
        datastore=datastore,
        args=args or logging_config,
        run_name=run_name,
        run_dir=run_dir,
        logger_type=logging_config.logger,
        logger_project=logging_config.logger_project,
        wandb_id=logging_config.wandb_id,
    )

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
        max_epochs=train_config.epochs,
        deterministic=True,
        default_root_dir=run_dir,
        strategy="auto",
        accelerator=device_name,
        num_nodes=compute_config.num_nodes,
        devices=devices,
        logger=training_logger,
        log_every_n_steps=1,
        callbacks=[val_checkpoint, latest_checkpoint],
        check_val_every_n_epoch=train_config.val_interval,
        precision=cast(Any, compute_config.precision),
        num_sanity_val_steps=train_config.num_sanity_val_steps,
    )

    if trainer.global_rank == 0:
        utils.init_training_logger_metrics(
            training_logger, val_steps=train_config.val_steps_to_log
        )

    if train_config.eval:
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=train_config.load,
        )
        checkpoint_path = train_config.load
    else:
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=train_config.load,
        )
        checkpoint_path = val_checkpoint.best_model_path or os.path.join(
            run_dir, "checkpoints", "min_val_loss.ckpt"
        )

    # Standard library
    from pathlib import Path

    # Local
    from .api import Run

    return Run(
        run_dir=Path(run_dir),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
    )


def run(
    args: Namespace,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
) -> Any:
    """Run the training or evaluation loop from parsed CLI arguments."""
    var_leads = getattr(args, "var_leads_metrics_watch", "{}")
    if isinstance(var_leads, str):
        var_leads = {int(k): v for k, v in json.loads(var_leads).items()}

    devices_list = getattr(args, "devices", ["auto"])

    model_config = ModelConfig(
        model=args.model,
        graph=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        processor_layers=args.processor_layers,
        mesh_aggr=args.mesh_aggr,
        output_std=args.output_std,
        g2m_gnn_type=getattr(args, "g2m_gnn_type", "InteractionNet"),
        m2g_gnn_type=getattr(args, "m2g_gnn_type", "InteractionNet"),
        mesh_up_gnn_type=getattr(args, "mesh_up_gnn_type", "InteractionNet"),
        mesh_down_gnn_type=getattr(
            args, "mesh_down_gnn_type", "InteractionNet"
        ),
    )
    train_config = TrainRunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        loss=args.loss,
        lr=args.lr,
        val_interval=args.val_interval,
        num_sanity_val_steps=args.num_sanity_val_steps,
        val_steps_to_log=args.val_steps_to_log,
        train_steps_to_log=args.train_steps_to_log,
        metrics_watch=args.metrics_watch,
        var_leads_metrics_watch=var_leads,
        load=args.load,
        restore_opt=args.restore_opt,
        eval=args.eval,
        n_example_pred=args.n_example_pred,
        create_gif=args.create_gif,
    )
    data_config = DataConfig(
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        num_workers=args.num_workers,
        load_single_member=args.load_single_member,
    )
    compute_config = ComputeConfig(
        seed=args.seed,
        num_nodes=args.num_nodes,
        devices=devices_list,
        precision=args.precision,
    )
    logging_config = LoggingConfig(
        logger=args.logger,
        logger_project=args.logger_project,
        logger_run_name=args.logger_run_name,
        runs_root=args.runs_root,
        wandb_id=args.wandb_id,
    )
    return fit(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        compute_config=compute_config,
        logging_config=logging_config,
        config=config,
        datastore=datastore,
        config_path=args.config_path,
        args=args,
    )


@logger.catch
def main(input_args: list[str] | None = None) -> None:
    """Main function for training and evaluating models."""
    parser = build_parser()
    args = parser.parse_args(input_args)
    run(args)


if __name__ == "__main__":
    main()
