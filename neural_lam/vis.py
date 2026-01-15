# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore
from .datastore.mike import MIKEDatastore


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(
    errors, datastore: BaseRegularGridDatastore | MIKEDatastore, title=None
):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons

    Args:
        errors (torch.Tensor): (d_f, pred_steps) tensor of errors
        datastore (BaseRegularGridDatastore): Datastore object
        title (str): Title of the plot

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object

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


def plot_on_axis(
    ax,
    da,
    datastore,
    vmin=None,
    vmax=None,
    ax_title=None,
    cmap="plasma",
):
    """
    Plot weather state on given axis

    Args:
        ax (matplotlib.axes.Axes): Axis object
        da (xr.DataArray): DataArray to plot
        datastore (BaseRegularGridDatastore): Datastore object
        vmin (float): Minimum value for colorbar
        vmax (float): Maximum value for colorbar
        ax_title (str): Title of the axis
        cmap (str): Colormap to use

    Returns:
        matplotlib.collections.QuadMesh: QuadMesh object
    """
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)

    gl = ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False
    )
    gl.top_labels = False
    gl.right_labels = False

    lats_lons = datastore.get_lat_lon("state")
    # If datastore provides a regular grid shape, use pcolormesh.
    # Otherwise (e.g. MIKEDatastore) fall back to an unstructured scatter plot.
    if (
        hasattr(datastore, "grid_shape_state")
        and getattr(datastore, "grid_shape_state") is not None
    ):
        grid_shape = (
            datastore.grid_shape_state.x,
            datastore.grid_shape_state.y,
        )
        lons = lats_lons[:, 0].reshape(grid_shape)
        lats = lats_lons[:, 1].reshape(grid_shape)
        data_to_plot = da.values.reshape(grid_shape)
        im = ax.pcolormesh(
            lons,
            lats,
            data_to_plot,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading="auto",
        )
    else:
        # Unstructured datastore (MIKE): use scatter plot
        lons = lats_lons[:, 0]
        lats = lats_lons[:, 1]
        data_vals = da.values.flatten()
        im = ax.scatter(
            lons,
            lats,
            c=data_vals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            s=10,
            marker="s",
            linewidths=0,
            alpha=0.9,
        )

    if ax_title:
        ax.set_title(ax_title, size=15)

    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore | MIKEDatastore,
    da_prediction: xr.DataArray = None,
    da_target: xr.DataArray = None,
    title=None,
    vrange=None,
):
    """
    Plot example prediction and ground truth with proper map projection.

    Args:
        datastore (BaseRegularGridDatastore): Datastore object
        da_prediction (xr.DataArray): Prediction to plot
        da_target (xr.DataArray): Ground truth to plot
        title (str): Title of the plot
        vrange (tuple): Range of values for colorbar

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    if vrange is None:
        vmin = min(da_prediction.min(), da_target.min())
        vmax = max(da_prediction.max(), da_target.max())
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": datastore.coords_projection},
    )

    for ax, da, subtitle in zip(
        axes, (da_target, da_prediction), ("Ground Truth", "Prediction")
    ):
        plot_on_axis(ax, da, datastore, vmin, vmax, subtitle, cmap="viridis")

    if title:
        fig.suptitle(title, size=20)

    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    fig.colorbar(axes[0].collections[0], cax=cbar_ax, orientation="horizontal")

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error,
    datastore: BaseRegularGridDatastore | MIKEDatastore,
    title=None,
    vrange=None,
):
    """
    Plot errors over spatial map

    Args:
        error (torch.Tensor): Error tensor
        datastore (BaseRegularGridDatastore): Datastore object
        title (str): Title of the plot
        vrange (tuple): Range of values for colorbar

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": datastore.coords_projection},
    )

    # For regular grids, reshape into 2D grid; otherwise pass 1D values
    if (
        hasattr(datastore, "grid_shape_state")
        and getattr(datastore, "grid_shape_state") is not None
    ):
        error_grid = (
            error.reshape(
                [
                    datastore.grid_shape_state.x,
                    datastore.grid_shape_state.y,
                ]
            )
            .cpu()
            .numpy()
        )
        da_to_plot = xr.DataArray(error_grid)
    else:
        # Unstructured: flatten values and let plot_on_axis scatter them
        da_to_plot = xr.DataArray(error.cpu().numpy().flatten())

    im = plot_on_axis(ax, da_to_plot, datastore, vmin, vmax, cmap="OrRd")

    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
