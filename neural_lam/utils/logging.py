"""Setup and configuration of training loggers (WandB / MLFlow)."""

# Standard library
import dataclasses
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..datastore.base import BaseDatastore

# Third-party
import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

# Local
from ..custom_loggers import CustomMLFlowLogger


@rank_zero_only
def log_on_rank_zero(
    msg: str, level: str = "info", *args: Any, **kwargs: Any
) -> None:
    """
    Log a message only on rank zero using loguru logger.

    Parameters
    ----------
    msg : str
        The message to log.
    level : str, default "info"
        The logging level (e.g. "info", "warning", "error").
    *args : Any
        Positional arguments passed to the logger.
    **kwargs : Any
        Keyword arguments passed to the logger.
    """
    if rank_zero_only.rank == 0:  # ty: ignore[unresolved-attribute]
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg, *args, **kwargs)


def init_training_logger_metrics(
    training_logger: Any, val_steps: list[int]
) -> None:
    """
    Configure validation metric aggregation for the active training logger.

    Parameters
    ----------
    training_logger : Any
        Logger instance used during training.
    val_steps : list of int
        Autoregressive rollout lengths to log as separate metrics.
    """
    experiment = training_logger.experiment
    if isinstance(training_logger, WandbLogger):
        experiment.define_metric("val_mean_loss", summary="min")
        for step in val_steps:
            experiment.define_metric(f"val_loss_unroll{step}", summary="min")
    elif isinstance(training_logger, MLFlowLogger):
        pass
    else:
        warnings.warn(
            "Only WandbLogger & MLFlowLogger is supported for tracking metrics.\
             Experiment results will only go to stdout."
        )


@rank_zero_only
def setup_training_logger(
    datastore: "BaseDatastore",
    args: Any | None = None,
    run_name: str = "",
    run_dir: str = "",
    *,
    logger_type: str = "wandb",
    logger_project: str = "neural_lam",
    wandb_id: str | None = None,
    config_dict: dict[str, Any] | None = None,
) -> pl.loggers.Logger:
    """
    Set up the training logger (WandB or MLFlow).

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore providing metadata for logging configuration.
    args : Any or None, optional
        Legacy parsed training arguments or ``LoggingConfig``.
    run_name : str, default ""
        Name of the run.
    run_dir : str, default ""
        Directory under which all artifacts for this run are written
        (logger ``save_dir``, checkpoints, Lightning ``default_root_dir``).
        Typically ``runs/<run_name>``.
    logger_type : str, default "wandb"
        Logger backend type ("wandb" or "mlflow").
    logger_project : str, default "neural_lam"
        Project name for the logger.
    wandb_id : str or None, optional
        W&B run ID for resuming experiments.
    config_dict : dict of {str: Any} or None, optional
        Dictionary of hyperparameters to log.

    Returns
    -------
    pl.loggers.Logger
        The initialized logger object.

    Raises
    ------
    ValueError
        If logger type is not ``'wandb'`` or ``'mlflow'``.
    """
    if args is not None:
        logger_type = getattr(args, "logger", logger_type)
        logger_project = getattr(
            args, "logger_project", getattr(args, "project", logger_project)
        )
        wandb_id = getattr(args, "wandb_id", wandb_id)
        if config_dict is None:
            if hasattr(args, "__dict__"):
                config_dict = vars(args)
            elif dataclasses.is_dataclass(args):
                config_dict = dataclasses.asdict(args)

    if config_dict is None:
        config_dict = {}

    if wandb_id and logger_type != "wandb":
        logger.warning(
            f"--wandb_id is set but logger is {logger_type!r}; "
            "the wandb_id will have no effect."
        )

    if logger_type == "wandb":
        wandb_resume = "allow" if wandb_id else None
        logger.info(f"Wandb resume mode: {wandb_resume!r} (id: {wandb_id!r})")
        return pl.loggers.WandbLogger(
            project=logger_project,
            name=None if wandb_id else run_name,
            config=dict(training=config_dict, datastore=datastore.config),
            resume=wandb_resume,
            id=wandb_id,
            save_dir=run_dir,
        )
    elif logger_type == "mlflow":
        if wandb_id is not None:
            warnings.warn(
                "--wandb_id is only used with --logger=wandb and will be "
                "ignored."
            )
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        training_logger = CustomMLFlowLogger(
            experiment_name=logger_project,
            tracking_uri=url,
            run_name=run_name,
            save_dir=run_dir,
        )
        training_logger.log_hyperparams(
            dict(training=config_dict, datastore=datastore.config)
        )
        return training_logger
    else:
        raise ValueError(
            f"Unsupported logger type: {logger_type!r}. "
            "Supported loggers are: 'wandb', 'mlflow'."
        )
