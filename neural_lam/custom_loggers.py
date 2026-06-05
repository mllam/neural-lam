# Standard library
import os
from typing import Optional

# Third-party
import matplotlib.pyplot as plt
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

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        run_name: str,
    ) -> None:
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        mlflow.start_run(run_id=self.run_id, log_system_metrics=True)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("run_id", self.run_id)

    @property
    def save_dir(self) -> str:
        """
        Returns the directory where the MLFlow artifacts are saved.
        Used to define the path to save output when using the logger.

        Returns
        -------
        str
            Path to the directory where the artifacts are saved.
        """
        return "mlruns"

    def log_image(
        self,
        key: str,
        images: list[plt.Figure],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a matplotlib figure as an image to MLFlow.

        Parameters
        ----------
        key : str
            Key to log the image under. If ``step`` is given, the actual
            key used is ``f"{key}_{step}"``.
        images : list of matplotlib.figure.Figure
            Figures to log; only the first element is used.
        step : int or None, optional
            Step to associate with the log entry. ``None`` logs without
            a step suffix.
        """
        # Third-party
        from botocore.exceptions import NoCredentialsError
        from PIL import Image

        if step is not None:
            key = f"{key}_{step}"

        # Need to save the image to a temporary file, then log that file
        # mlflow.log_image, should do this automatically, but is buggy
        temporary_image = f"{key}.png"
        try:
            images[0].savefig(temporary_image)
            with Image.open(temporary_image) as img:
                mlflow.log_image(img, f"{key}.png")
        except NoCredentialsError:
            logger.error("Error logging image\nSet AWS credentials")
            raise
        finally:
            if os.path.exists(temporary_image):
                os.remove(temporary_image)
