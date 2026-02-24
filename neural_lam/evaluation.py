# Standard library
import os
import warnings
from typing import Any, Dict, List

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# Local
from . import vis
from .utils import get_integer_time
from .weather_dataset import WeatherDataset


def create_dataarray_from_tensor(
    tensor: torch.Tensor,
    time: np.ndarray,
    split: str,
    category: str,
    datastore: Any,
) -> xr.DataArray:
    """
    Create an `xr.DataArray` from a tensor, with the correct dimensions and
    coordinates to match the datastore used by the model. This function in
    in effect is the inverse of what is returned by
    `WeatherDataset.__getitem__`.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to convert to a `xr.DataArray` with dimensions [time,
        grid_index, feature]. The tensor will be copied to the CPU if it is
        not already there.
    time : np.ndarray
        The time index or indices for the data.
    split : str
        The split of the data, either 'train', 'val', or 'test'
    category : str
        The category of the data, either 'state' or 'forcing'
    datastore : BaseDatastore
        The datastore object used.
    """
    weather_dataset = WeatherDataset(datastore=datastore, split=split)
    da = weather_dataset.create_dataarray_from_tensor(
        tensor=tensor, time=time, category=category
    )
    return da


def plot_examples(
    datastore: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    n_examples: int,
    split: str,
    plotted_examples: int,
    state_std: torch.Tensor,
    state_mean: torch.Tensor,
    save_dir: str,
    logger: Any,
) -> int:
    """
    Plot the first n_examples forecasts from batch

    datastore: datastore object
    prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
    target: (B, pred_steps, num_grid_nodes, d_f), target.
    time: tensor with times
    n_examples: number of forecasts to plot
    split: split info
    plotted_examples: already plotted example count
    state_std: standard deviation buffer for state
    state_mean: mean buffer for state
    save_dir: directory to save `.pt` example slices
    logger: PyTorch Lightning logger
    """
    # Rescale to original data scale
    prediction_rescaled = prediction * state_std + state_mean
    target_rescaled = target * state_std + state_mean

    time_step_int, time_step_unit = get_integer_time(datastore.step_length)

    # Iterate over the examples
    for pred_slice, target_slice, time_slice in zip(
        prediction_rescaled[:n_examples],
        target_rescaled[:n_examples],
        time[:n_examples],
    ):
        # Each slice is (pred_steps, num_grid_nodes, d_f)
        plotted_examples += 1  # Increment already here

        time_slice_np = np.array(time_slice.cpu(), dtype="datetime64[ns]")

        da_prediction = create_dataarray_from_tensor(
            tensor=pred_slice,
            time=time_slice_np,
            split=split,
            category="state",
            datastore=datastore,
        ).unstack("grid_index")
        da_target = create_dataarray_from_tensor(
            tensor=target_slice,
            time=time_slice_np,
            split=split,
            category="state",
            datastore=datastore,
        ).unstack("grid_index")

        var_vmin = (
            torch.minimum(
                pred_slice.flatten(0, 1).min(dim=0)[0],
                target_slice.flatten(0, 1).min(dim=0)[0],
            )
            .cpu()
            .numpy()
        )  # (d_f,)
        var_vmax = (
            torch.maximum(
                pred_slice.flatten(0, 1).max(dim=0)[0],
                target_slice.flatten(0, 1).max(dim=0)[0],
            )
            .cpu()
            .numpy()
        )  # (d_f,)
        var_vranges = list(zip(var_vmin, var_vmax))

        # Iterate over prediction horizon time steps
        for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
            # Create one figure per variable at this time step
            var_figs = [
                vis.plot_prediction(
                    datastore=datastore,
                    title=f"{var_name} ({var_unit}), "
                    f"t={t_i} ({(time_step_int * t_i)}"
                    f"{time_step_unit})",
                    vrange=var_vrange,
                    da_prediction=da_prediction.isel(
                        state_feature=var_i, time=t_i - 1
                    ).squeeze(),
                    da_target=da_target.isel(
                        state_feature=var_i, time=t_i - 1
                    ).squeeze(),
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(
                        datastore.get_vars_names("state"),
                        datastore.get_vars_units("state"),
                        var_vranges,
                    )
                )
            ]

            example_i = plotted_examples

            for var_name, fig in zip(
                datastore.get_vars_names("state"), var_figs
            ):

                # We need treat logging images differently for different
                # loggers. WANDB can log multiple images to the same key,
                # while other loggers, as MLFlow, need unique keys for
                # each image.
                if isinstance(logger, pl.loggers.WandbLogger):
                    key = f"{var_name}_example_{example_i}"
                else:
                    key = f"{var_name}_example"

                if hasattr(logger, "log_image"):
                    logger.log_image(key=key, images=[fig], step=t_i)
                else:
                    warnings.warn(
                        f"{logger} does not support image logging."
                    )

            plt.close(
                "all"
            )  # Close all figs for this time step, saves memory

        # Save pred and target as .pt files
        torch.save(
            pred_slice.cpu(),
            os.path.join(
                save_dir,
                f"example_pred_{plotted_examples}.pt",
            ),
        )
        torch.save(
            target_slice.cpu(),
            os.path.join(
                save_dir,
                f"example_target_{plotted_examples}.pt",
            ),
        )

    return plotted_examples


def create_metric_log_dict(
    metric_tensor: torch.Tensor,
    prefix: str,
    metric_name: str,
    datastore: Any,
    save_dir: str,
    metrics_watch: List[str],
    var_leads_metrics_watch: Dict[int, List[int]],
) -> Dict[str, Any]:
    """
    Put together a dict with everything to log for one metric. Also saves
    plots as pdf and csv if using test prefix.

    metric_tensor: (pred_steps, d_f), metric values per time and variable
    prefix: string, prefix to use for logging metric_name: string, name of
    the metric
    datastore: Datastore object for extracting variables
    save_dir: Directory where the csv and pdf are saved for test split
    metrics_watch: The metrics being monitored explicitly
    var_leads_metrics_watch: Lead-variables that are being checked

    Return: log_dict: dict with everything to log for given metric
    """
    log_dict = {}
    metric_fig = vis.plot_error_map(
        errors=metric_tensor,
        datastore=datastore,
    )
    full_log_name = f"{prefix}_{metric_name}"
    log_dict[full_log_name] = metric_fig

    if prefix == "test":
        # Save pdf
        metric_fig.savefig(
            os.path.join(save_dir, f"{full_log_name}.pdf")
        )
        # Save errors also as csv
        np.savetxt(
            os.path.join(save_dir, f"{full_log_name}.csv"),
            metric_tensor.cpu().numpy(),
            delimiter=",",
        )

    # Check if metrics are watched, log exact values for specific vars
    var_names = datastore.get_vars_names(category="state")
    if full_log_name in metrics_watch:
        for var_i, timesteps in var_leads_metrics_watch.items():
            var_name = var_names[var_i]
            for step in timesteps:
                key = f"{full_log_name}_{var_name}_step_{step}"
                log_dict[key] = metric_tensor[step - 1, var_i]

    return log_dict


def evaluate_metrics(
    metrics_dict: Dict[str, List[torch.Tensor]],
    prefix: str,
    state_std: torch.Tensor,
    datastore: Any,
    save_dir: str,
    metrics_watch: List[str],
    var_leads_metrics_watch: Dict[int, List[int]],
    all_gather_cat_fn: Any,
    is_global_zero: bool,
) -> Dict[str, Any]:
    """
    Aggregate and create error map plots for all metrics in metrics_dict

    metrics_dict: dictionary with metric_names and list of tensors
        with step-evals.
    prefix: string, prefix to use for logging
    state_std: standard deviations for state variables
    datastore: Datastore object for setting vars
    save_dir: Path to directory where test plots are output
    metrics_watch: Monitored metric lists
    var_leads_metrics_watch: Variable lists checked
    all_gather_cat_fn: Distributing function to gather tensors across ranks
    is_global_zero: Checked for gathering rank root
    """
    log_dict = {}
    for metric_name, metric_val_list in metrics_dict.items():
        metric_tensor = all_gather_cat_fn(
            torch.cat(metric_val_list, dim=0)
        )  # (N_eval, pred_steps, d_f)

        if is_global_zero:
            metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
            # (pred_steps, d_f)

            # Take square root after all averaging to change MSE to RMSE
            if "mse" in metric_name:
                metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                metric_name = metric_name.replace("mse", "rmse")

            # NOTE: we here assume rescaling for all metrics is linear
            metric_rescaled = metric_tensor_averaged * state_std
            # (pred_steps, d_f)
            log_dict.update(
                create_metric_log_dict(
                    metric_tensor=metric_rescaled,
                    prefix=prefix,
                    metric_name=metric_name,
                    datastore=datastore,
                    save_dir=save_dir,
                    metrics_watch=metrics_watch,
                    var_leads_metrics_watch=var_leads_metrics_watch,
                )
            )

    # Ensure that log_dict has structure for
    # logging as dict(str, plt.Figure) or dict(str, float/torch.Tensor)
    assert all(
        isinstance(key, str)
        and (
            isinstance(value, plt.Figure)
            or isinstance(value, (torch.Tensor, float, int))
        )
        for key, value in log_dict.items()
    )

    return log_dict
