"""Custom logging utilities (e.g., MLFlow wrappers) used in Neural-LAM."""

# Standard library
import sys

# Third-party
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from loguru import logger


class CustomMLFlowLogger(pl.loggers.MLFlowLogger):
    """
    Custom MLFlow logger that adds the `log_image()` functionality not
    present in the default implementation from pytorch-lightning as
    of version `2.0.3` at least.
    """

    def __init__(self, experiment_name, tracking_uri, run_name):
        """
        Initialize the logger and start an MLflow run.

        Parameters
        ----------
        experiment_name : str
            Target MLflow experiment.
        tracking_uri : str
            MLflow tracking server URI.
        run_name : str
            Human-readable run name stored as ``mlflow.runName``.
        """
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        mlflow.start_run(run_id=self.run_id, log_system_metrics=True)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("run_id", self.run_id)

    @property
    def save_dir(self):
        """
        Returns the directory where the MLFlow artifacts are saved.
        Used to define the path to save output when using the logger.

        Returns
        -------
        str
            Path to the directory where the artifacts are saved.
        """
        return "mlruns"

    def log_image(self, key, images, step=None):
        """
        Log one or more Matplotlib figures as images in MLflow.

        Parameters
        ----------
        key : str
            Identifier under which to log the image.
        images : Sequence[matplotlib.figure.Figure]
            Figures to export; only the first element is logged.
        step : int or None, optional
            Optional training step index appended to ``key``.

        Raises
        ------
        SystemExit
            If AWS credentials for the MLflow artifact store are missing.
        """
        # Third-party
        from botocore.exceptions import NoCredentialsError
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
        except NoCredentialsError:
            logger.error("Error logging image\nSet AWS credentials")
            sys.exit(1)
