# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


def plot_on_axis(
    ax,
    da,
    datastore,
    vmin=None,
    vmax=None,
    ax_title=None,
    cmap="plasma",
    boundary_alpha=None,
    crop_to_interior=False,
):
    """Plot weather state on given axis using datastore metadata.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on. Should have a cartopy projection.
    da : xarray.DataArray or np.ndarray
        The data to plot. Should have shape (N_grid,).
    datastore : BaseRegularGridDatastore
        The datastore containing metadata about the grid.
    vmin : float, optional
        Minimum value for color scale.
    vmax : float, optional
        Maximum value for color scale.
    ax_title : str, optional
        Title for the axis.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for plotting.
    boundary_alpha : float, optional
        If provided, overlay boundary mask with given alpha transparency.
    crop_to_interior : bool, optional
        If True, crop the plot to the interior region.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh object created by pcolormesh.

    """

    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False

    lats_lons = datastore.get_lat_lon("state")
    grid_shape = (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    lons = lats_lons[:, 0].reshape(grid_shape)
    lats = lats_lons[:, 1].reshape(grid_shape)

    if isinstance(da, xr.DataArray) and "x" in da.dims and "y" in da.dims:
        da = da.transpose("x", "y")

    values = np.asarray(getattr(da, "values", da)).reshape(grid_shape)

    mesh = ax.pcolormesh(
        lons,
        lats,
        values,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        shading="auto",
    )

    if boundary_alpha is not None:
        # Overlay boundary mask
        mask_da = datastore.boundary_mask
        mask_values = mask_da.values
        if mask_values.ndim == 2 and mask_values.shape[1] == 1:
            mask_values = mask_values[:, 0]
        mask_2d = mask_values.reshape(grid_shape)

        # Create overlay: 1 where boundary, NaN where interior
        overlay = np.where(mask_2d == 1, 1.0, np.nan)

        ax.pcolormesh(
            lons,
            lats,
            overlay,
            transform=ccrs.PlateCarree(),
            cmap=matplotlib.colors.ListedColormap([(1, 1, 1, boundary_alpha)]),
            shading="auto",
        )

    if crop_to_interior:
        # Calculate extent of interior
        mask_da = datastore.boundary_mask
        mask_values = mask_da.values
        if mask_values.ndim == 2 and mask_values.shape[1] == 1:
            mask_values = mask_values[:, 0]
        mask_2d = mask_values.reshape(grid_shape)

        interior_points = mask_2d == 0
        if np.any(interior_points):
            interior_lons = lons[interior_points]
            interior_lats = lats[interior_points]

            min_lon, max_lon = interior_lons.min(), interior_lons.max()
            min_lat, max_lat = interior_lats.min(), interior_lats.max()

            ax.set_extent(
                [min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree()
            )

    if ax_title:
        ax.set_title(ax_title, size=15)

    return mesh


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
    datastore: BaseRegularGridDatastore,
    da_prediction: xr.DataArray = None,
    da_target: xr.DataArray = None,
    title=None,
    vrange=None,
    boundary_alpha=0.7,
    crop_to_interior=True,
):
    """
    Plot example prediction and grond truth.

    Each has shape (N_grid,)

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
        plot_on_axis(
            ax=ax,
            da=da,
            datastore=datastore,
            vmin=vmin,
            vmax=vmax,
            ax_title=subtitle,
            cmap="viridis",
            boundary_alpha=boundary_alpha,
            crop_to_interior=crop_to_interior,
        )

    if title:
        fig.suptitle(title, size=20)

    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    fig.colorbar(axes[0].collections[0], cax=cbar_ax, orientation="horizontal")

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error,
    datastore: BaseRegularGridDatastore,
    title=None,
    vrange=None,
    boundary_alpha=0.7,
    crop_to_interior=True,
):
    """Plot spatial error with projection-aware axes."""

    grid_shape = [datastore.grid_shape_state.x, datastore.grid_shape_state.y]

    value_source = error
    if hasattr(value_source, "detach"):
        value_source = value_source.detach()
    if hasattr(value_source, "cpu"):
        value_source = value_source.cpu()

    error_np = np.asarray(value_source)

    if vrange is None:
        vmin = float(np.nanmin(error_np))
        vmax = float(np.nanmax(error_np))
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": datastore.coords_projection},
    )

    error_grid = error_np.reshape(grid_shape)

    mesh = plot_on_axis(
        ax=ax,
        da=xr.DataArray(error_grid),
        datastore=datastore,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
        boundary_alpha=boundary_alpha,
        crop_to_interior=crop_to_interior,
    )

    cbar = fig.colorbar(mesh, ax=ax, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
