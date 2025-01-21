# Standard library
import json
import random
import sys
import time
from argparse import ArgumentParser

# Third-party
import mlflow

# for logging the model:
import mlflow.pytorch
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastore
from .models import GraphLAM, HiLAM, HiLAMParallel
from .weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


class CustomMLFlowLogger(pl.loggers.MLFlowLogger):
    """
    Custom MLFlow logger that adds functionality not present in the default
    """

    def __init__(self, experiment_name, tracking_uri):
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )
        mlflow.start_run(run_id=self.run_id, log_system_metrics=True)
        mlflow.log_param("run_id", self.run_id)

    @property
    def save_dir(self):
        """
        Returns the directory where the MLFlow artifacts are saved
        """
        return "mlruns"

    def log_image(self, key, images, step=None):
        """
        Log a matplotlib figure as an image to MLFlow

        key: str
            Key to log the image under
        images: list
            List of matplotlib figures to log
        step: Union[int, None]
            Step to log the image under. If None, logs under the key directly
        """
        # Third-party
        import botocore
        from PIL import Image

        if step is not None:
            key = f"{key}_{step}"

        # Need to save the image to a temporary file, then log that file
        # mlflow.log_image, should do this automatically, but is buggy
        temporary_image = f"{key}.png"
        images[0].savefig(temporary_image)

        img = Image.open(temporary_image)
        try:
            mlflow.log_image(img, f"{key}.png")
        except botocore.exceptions.NoCredentialsError:
            logger.error("Error logging image\nSet AWS credentials")
            sys.exit(1)


def _setup_training_logger(config, datastore, args, run_name):
    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            project=args.logger_project,
            name=run_name,
            config=dict(training=vars(args), datastore=datastore._config),
        )
    elif args.logger == "mlflow":
        url = args.logger_url
        if url is None:
            raise ValueError(
                "MLFlow logger requires a URL to the MLFlow server"
            )
        logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
        )
        logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return logger


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
        help="If optimizer state should be restored with model "
        "(default: false)",
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
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
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
        "--save_predictions",
        action="store_true",
        help="If predictions should be saved to disk as a zarr dataset "
        "(default: false)",
    )

    # Logger Settings
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        help="Logger to use for training (wandb/mlflow) (default: wandb)",
    )
    parser.add_argument(
        "--logger-url",
        type=str,
        default=None,
        help="URL to the logger server (default: None)",
    )
    parser.add_argument(
        "--logger-project",
        type=str,
        default="neural_lam",
        help="Logger project name, for eg. Wandb (default: neural_lam)",
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
        "--num_past_forcing_steps",
        type=int,
        default=1,
        help="Number of past time steps to use as input for forcing data",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=1,
        help="Number of future time steps to use as input for forcing data",
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

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore to use
    config, datastore = load_config_and_datastore(config_path=args.config_path)

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
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
    model = ModelClass(args, config=config, datastore=datastore)

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    training_logger = _setup_training_logger(
        config=config, datastore=datastore, args=args, run_name=run_name
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        # devices=4,
        devices=[0, 3],
        # devices=[0, 1, 2],
        # strategy="auto",
        # devices=1,  # For eval mode
        # num_nodes=1,  # For eval mode
        accelerator=device_name,
        logger=training_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
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
