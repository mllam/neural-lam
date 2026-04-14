# Standard library
import os
import warnings

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, datastore: BaseRegularGridDatastore, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    time_step_int, time_step_unit = utils.get_integer_time(step_length)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1
    pred_hor_h = time_step_int * pred_hor_i
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel(f"Lead time ({time_step_unit[0]})", size=label_size)

    ax.set_yticks(np.arange(d_f))
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    y_ticklabels = [
        f"{name} ({unit})" for name, unit in zip(var_names, var_units)
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore,
    da_prediction: xr.DataArray,
    da_target: xr.DataArray,
    title=None,
    vrange=None,
):
    """
    Plot example prediction and grond truth.

    Each has shape (N_grid,)

    """
    # Get common scale for values
    if vrange is None:
        vmin = min(da_prediction.min(), da_target.min())
        vmax = max(da_prediction.max(), da_target.max())
    elif vrange is not None:
        vmin, vmax = vrange

    extent = datastore.get_xy_extent("state")

    # Set up masking of border region
    da_mask = datastore.unstack_grid_coords(datastore.boundary_mask)
    mask_values = np.invert(da_mask.values.astype(bool)).astype(float)
    pixel_alpha = mask_values.clip(0.7, 1)  # Faded border region

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": datastore.coords_projection},
    )

    # Plot pred and target
    for ax, da in zip(axes, (da_target, da_prediction)):
        ax.coastlines()  # Add coastline outlines
        da.plot.imshow(
            ax=ax,
            origin="lower",
            x="x",
            extent=extent,
            alpha=pixel_alpha.T,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
            transform=datastore.coords_projection,
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error, datastore: BaseRegularGridDatastore, title=None, vrange=None
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    extent = datastore.get_xy_extent("state")

    # Set up masking of border region
    da_mask = datastore.unstack_grid_coords(datastore.boundary_mask)
    mask_reshaped = da_mask.values
    pixel_alpha = mask_reshaped.clip(0.7, 1)  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": datastore.coords_projection},
    )

    ax.coastlines()  # Add coastline outlines
    error_grid = (
        error.reshape(
            [datastore.grid_shape_state.x, datastore.grid_shape_state.y]
        )
        .T.cpu()
        .numpy()
    )

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=extent,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


def plot_examples(
    datastore: BaseRegularGridDatastore,
    logger: pl.loggers.Logger,
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_batch: torch.Tensor,
    first_example_idx: int = 0,
) -> None:
    """
    Plot example forecasts from provided tensors.

    Args:
        datastore: The object containing dataset metadata.
        logger: The logger instance used to save the images.
        prediction: Output tensors predicted from the model.
        target: Ground truth tensors.
        time_batch: Time timestamps corresponding to the data.
        first_example_idx: Starting index for naming saved files/logs.
    """
    time_step_int, time_step_unit = utils.get_integer_time(
        datastore.step_length
    )

    for i, (pred_slice, target_slice, time_slice) in enumerate(
        zip(prediction, target, time_batch)
    ):
        example_i = first_example_idx + i

        # Detach tensors to safely separate from autograd graph
        pred_slice = pred_slice.detach()
        target_slice = target_slice.detach()

        time_arr = np.array(time_slice.cpu(), dtype="datetime64[ns]")

        da_prediction = datastore.create_dataarray_from_tensor(
            tensor=pred_slice, time=time_arr, category="state"
        ).unstack("grid_index")

        da_target = datastore.create_dataarray_from_tensor(
            tensor=target_slice, time=time_arr, category="state"
        ).unstack("grid_index")

        var_vmin = (
            torch.minimum(
                pred_slice.flatten(0, 1).min(dim=0)[0],
                target_slice.flatten(0, 1).min(dim=0)[0],
            )
            .cpu()
            .numpy()
        )
        var_vmax = (
            torch.maximum(
                pred_slice.flatten(0, 1).max(dim=0)[0],
                target_slice.flatten(0, 1).max(dim=0)[0],
            )
            .cpu()
            .numpy()
        )
        var_vranges = list(zip(var_vmin, var_vmax))

        for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
            var_figs = [
                plot_prediction(
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

            for var_name, fig in zip(
                datastore.get_vars_names("state"), var_figs
            ):
                if isinstance(logger, pl.loggers.WandbLogger):
                    key = f"{var_name}_example_{example_i}"
                else:
                    key = f"{var_name}_example"

                if hasattr(logger, "log_image"):
                    logger.log_image(key=key, images=[fig], step=t_i)
                else:
                    warnings.warn(f"{logger} does not support image logging.")

            plt.close("all")

        pred_filename = f"example_pred_{example_i}.pt"
        torch.save(
            pred_slice.cpu(),
            os.path.join(logger.save_dir, pred_filename),
        )

        target_filename = f"example_target_{example_i}.pt"
        torch.save(
            target_slice.cpu(),
            os.path.join(logger.save_dir, target_filename),
        )
