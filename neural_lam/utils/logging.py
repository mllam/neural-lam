"""Setup and configuration of training loggers (WandB / MLFlow)."""

# Standard library
import os
import warnings
from typing import Any

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
    if rank_zero_only.rank == 0:
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
    datastore: Any, args: Any, run_name: str, run_dir: str
) -> Any:
    """
    Set up the training logger (WandB or MLFlow).

    Parameters
    ----------
    datastore : Any
        Datastore providing metadata for logging configuration.
    args : argparse.Namespace
        Parsed training arguments controlling the logger backend.
    run_name : str
        Name of the run.

    run_dir : str
        Directory under which all artifacts for this run are written
        (logger ``save_dir``, checkpoints, Lightning ``default_root_dir``).
        Typically ``runs/<run_name>``.

    Returns
    -------
    Any
        The initialized logger object.

    Raises
    ------
    ValueError
        If ``args.logger`` is not ``'wandb'`` or ``'mlflow'``.

    Notes
    -----
    When ``--wandb_id`` is given, ``resume="allow"`` is set automatically:
    W&B resumes the run if it exists, or creates it with that ID otherwise.
    This allows the same job script to be safely resubmitted on HPC systems.
    The run name is set to ``None`` when resuming to preserve the existing name.
    """
    if args.wandb_id and args.logger != "wandb":
        logger.warning(
            f"--wandb_id is set but logger is {args.logger!r}; "
            "the wandb_id will have no effect."
        )

    if args.logger == "wandb":
        wandb_resume = "allow" if args.wandb_id else None
        logger.info(
            f"Wandb resume mode: {wandb_resume!r} (id: {args.wandb_id!r})"
        )
        return pl.loggers.WandbLogger(
            project=args.logger_project,
            name=None if args.wandb_id else run_name,
            config=dict(training=vars(args), datastore=datastore._config),
            resume=wandb_resume,
            id=args.wandb_id,
            save_dir=run_dir,
        )
    elif args.logger == "mlflow":
        if args.wandb_id is not None:
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
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
            save_dir=run_dir,
        )
        training_logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )
        return training_logger
    else:
        raise ValueError(
            f"Unsupported logger type: {args.logger!r}. "
            "Supported loggers are: 'wandb', 'mlflow'."
        )
