# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

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
    pred,
    target,
    datastore: BaseRegularGridDatastore,
    title=None,
    vrange=None,
):
    """
    Plot example prediction and grond truth.

    Each has shape (N_grid,)

    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    extent = datastore.get_xy_extent("state")

    # Set up masking of border region
    da_mask = datastore.unstack_grid_coords(datastore.boundary_mask)
    mask_reshaped = da_mask.values
    pixel_alpha = (
        #mask_reshaped.clamp(0.7, 1).cpu().numpy()
        mask_reshaped.clip(0.7, 1)
    )  # Faded border region

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": datastore.coords_projection},
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        data_grid = (
            data.reshape(list(datastore.grid_shape_state.values.values()))
            .cpu()
            .numpy()
        )
        im = ax.imshow(
            data_grid,
            origin="lower",
            extent=extent,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

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
    pixel_alpha = (
        #mask_reshaped.clamp(0.7, 1).cpu().numpy()
        mask_reshaped.clip(0.7, 1)
    )  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": datastore.coords_projection},
    )

    ax.coastlines()  # Add coastline outlines
    error_grid = (
        error.reshape(list(datastore.grid_shape_state.values.values()))
        .cpu()
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
