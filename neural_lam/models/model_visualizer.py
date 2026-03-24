# Standard library
import os
import warnings
from typing import Any, Dict

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

# First-party
from .. import vis
from ..datastore import BaseDatastore


class ModelVisualizer:
    """
    Handles visualization of model predictions and metrics.

    This class centralizes all plotting and visualization logic for weather models,
    including prediction plots, error maps, and spatial loss visualizations.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        args: Any,
        time_step_int: int,
        time_step_unit: str,
    ):
        """
        Initialize the ModelVisualizer.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore for accessing data metadata
        args : Any
            Arguments object containing configuration
        time_step_int : int
            Integer part of time step
        time_step_unit : str
            Unit of time step (e.g., 'h' for hours)
        """
        self.datastore = datastore
        self.args = args
        self.time_step_int = time_step_int
        self.time_step_unit = time_step_unit

    def create_metric_log_dict(
        self,
        metric_tensor: torch.Tensor,
        prefix: str,
        metric_name: str,
        logger: Any,
    ) -> Dict[str, Any]:
        """
        Put together a dict with everything to log for one metric.

        Also saves plots as pdf and csv if using test prefix.

        Parameters
        ----------
        metric_tensor : torch.Tensor
            Metric values per time and variable, shape (pred_steps, d_f)
        prefix : str
            Prefix to use for logging (e.g., 'val', 'test')
        metric_name : str
            Name of the metric
        logger : Any
            Logger instance for saving files

        Returns
        -------
        Dict[str, Any]
            Dictionary with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self.datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(logger.save_dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(logger.save_dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self.datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(
        self,
        metrics_dict: Dict[str, Any],
        prefix: str,
        state_std: torch.Tensor,
        logger: Any,
        trainer: Any,
    ):
        """
        Aggregate and create error map plots for all metrics in metrics_dict.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary with metric_names and list of tensors with step-evals
        prefix : str
            Prefix to use for logging
        state_std : torch.Tensor
            State standard deviation for rescaling
        logger : Any
            Logger instance
        trainer : Any
            Trainer instance for rank and sanity check info
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            # Gather metrics across devices
            metric_list_cat = torch.cat(metric_val_list, dim=0)
            # Note: gathering is handled by the caller (ARModel.all_gather_cat)

            if trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(
                    metric_list_cat, dim=0
                )  # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = (
                    metric_tensor_averaged * state_std
                )  # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name, logger
                    )
                )

        # Ensure that log_dict has structure for logging as dict(str, plt.Figure)
        assert all(
            isinstance(key, str) and isinstance(value, plt.Figure)
            for key, value in log_dict.items()
        )

        if trainer.is_global_zero and not trainer.sanity_checking:
            current_epoch = trainer.current_epoch

            for key, figure in log_dict.items():
                # For other loggers than wandb, add epoch to key.
                # Wandb can log multiple images to the same key, while other
                # loggers, such as MLFlow need unique keys for each image.
                if not isinstance(logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(logger, "log_image"):
                    logger.log_image(key=key, images=[figure])

            plt.close("all")  # Close all figs

    def plot_spatial_loss(
        self,
        spatial_loss_tensor: torch.Tensor,
        logger: Any,
        trainer: Any,
    ):
        """
        Plot spatial loss maps.

        Parameters
        ----------
        spatial_loss_tensor : torch.Tensor
            Spatial loss tensor, shape (N_test, N_log, num_grid_nodes)
        logger : Any
            Logger instance
        trainer : Any
            Trainer instance
        """
        if trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self.datastore,
                    title=f"Test loss, t={t_i} "
                    f"({(self.time_step_int * t_i)} {self.time_step_unit})",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same key, sequentially
            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(logger, "log_image"):
                    logger.log_image(key=key, images=[fig])

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(error=loss_map, datastore=self.datastore)
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(logger.save_dir, "spatial_loss_maps")
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(logger.save_dir, "mean_spatial_loss.pt"),
            )
