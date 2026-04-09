# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_heatmap(errors, datastore: BaseRegularGridDatastore, title=None,errors_norm=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    if errors_norm is not None:
        errors_norm_np = errors_norm.T.cpu().numpy()
        if errors_norm_np.ndim == 1:
         errors_norm_np = errors_norm_np.reshape(1, -1)
    else:
        errors_norm_np = None
    
    if errors_np.ndim == 1:
        errors_np = errors_np.reshape(1,-1)
    

    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    # Normalize all errors to [0,1] for color map
    
    if errors_norm_np is not None:
       vmin = np.min(errors_norm_np)
       vmax = np.max(errors_norm_np)
       color_data = errors_norm_np
    else:
        vmin = np.min(errors_np)
        vmax = np.max(errors_np)
        color_data = errors_np
    

    time_step_int, time_step_unit = utils.get_integer_time(step_length)

    fig_width = max(6, pred_steps*0.8)
    fig_height = max(4, d_f *0.6)

    fig, ax = plt.subplots(figsize=(fig_width,fig_height))

    ax.imshow(
        color_data,
        cmap="OrRd",
        vmin=vmin,
        vmax=vmax,
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
    n_cells = d_f * pred_steps
    show_annotations = n_cells <=200
    if show_annotations:
        for(j,i), error in np.ndenumerate(errors_np):
            formatted_error = f"{error:3f}" if error < 9999 else f"{error: .2E}"
            ax.text(i, j, formatted_error, ha="center", va="center")

    fontsize = max(6,min(14, 200//max(d_f , pred_steps)))
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1
    pred_hor_h = time_step_int * pred_hor_i
    ax.set_xticklabels(pred_hor_h, size=fontsize)
    ax.set_xlabel(f"Lead time ({time_step_unit[0]})", size=fontsize)

    ax.set_yticks(np.arange(d_f))
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    y_ticklabels = [
        f"{name} ({unit})" #for name, unit in zip(var_names, var_units)
        for name, unit in zip(var_names[:d_f], var_units[:d_f])
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=fontsize)

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
    errors_np = (
        error.reshape(
            [datastore.grid_shape_state.x, datastore.grid_shape_state.y]
        )
        .T.cpu()
        .numpy()
    )

    im = ax.imshow(
        errors_np,
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
