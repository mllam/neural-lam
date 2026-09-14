"""Custom logging utilities (e.g., MLFlow wrappers) used in Neural-LAM."""

# Standard library
import os

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
        save_dir: str,
    ) -> None:
        """
        Initialize the logger, ensure ``save_dir`` exists, and start the
        MLflow run.

        Parameters
        ----------
        experiment_name : str
            Target MLflow experiment.
        tracking_uri : str
            MLflow tracking server URI.
        run_name : str
            Human-readable run name stored as ``mlflow.runName``.
        save_dir : str
            Directory where ``log_image`` writes temporary figure files.
            Created eagerly with ``exist_ok=True``.

        Notes
        -----
        Starts the MLflow run with ``log_system_metrics=True`` and also
        records ``run_id`` as an MLflow param.
        """
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

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
        return self._save_dir

    def log_image(
        self,
        key: str,
        images: list[plt.Figure],
        step: int | None = None,
    ) -> None:
        """
        Log one or more matplotlib figures as images to MLFlow.

        When ``images`` contains more than one figure, each is logged under
        a key suffixed with its position in the list (``key_0``, ``key_1``,
        ...). A single-figure list is logged under the bare key.

        Parameters
        ----------
        key : str
            Key to log the image under. If ``step`` is given, ``_{step}``
            is appended before any per-figure index suffix.
        images : list of matplotlib.figure.Figure
            Figures to log.
        step : int or None, optional
            Step to associate with the log entry. ``None`` logs without
            a step suffix.

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

        # Need to save each figure to a temporary file, then log it
        # mlflow.log_image should do this automatically, but is buggy
        for i, fig in enumerate(images):
            img_key = f"{key}_{i}" if len(images) > 1 else key
            temporary_image = os.path.join(self.save_dir, f"{img_key}.png")
            try:
                fig.savefig(temporary_image)
                with Image.open(temporary_image) as img:
                    mlflow.log_image(img, f"{img_key}.png")
            except NoCredentialsError:
                logger.error("Error logging image\nSet AWS credentials")
                raise
            finally:
                if os.path.exists(temporary_image):
                    os.remove(temporary_image)
