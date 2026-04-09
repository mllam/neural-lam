# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore

# Font sizes shared across all plot functions for visual consistency.
_TITLE_SIZE = 13  # suptitle and per-axes titles
_LABEL_SIZE = 11  # axis / colorbar labels
_TICK_SIZE = 11  # tick labels


def _tex_safe(s: str) -> str:
    """Escape TeX special characters in s if TeX rendering is currently active.

    Needed because % is a TeX comment character; without escaping it would
    silently truncate any text that follows it (e.g. the title for r2m (%)).
    """
    if plt.rcParams.get("text.usetex", False):
        s = s.replace("%", r"\%")
    return s


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
    da : xarray.DataArray
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
    gl.xlabel_style = {"size": _TICK_SIZE}
    gl.ylabel_style = {"size": _TICK_SIZE}

    lats_lons = datastore.get_lat_lon("state")
    grid_shape = (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    lons = lats_lons[:, 0].reshape(grid_shape)
    lats = lats_lons[:, 1].reshape(grid_shape)

    if isinstance(da, xr.DataArray) and "x" in da.dims and "y" in da.dims:
        da = da.transpose("x", "y")

    values = da.values.reshape(grid_shape)

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
        ax.set_title(ax_title, size=_TITLE_SIZE)

    return mesh


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
        ax.set_title(title, size=_TITLE_SIZE)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore,
    da_prediction: xr.DataArray,
    da_target: xr.DataArray,
    title=None,
    vrange=None,
    boundary_alpha=0.7,
    crop_to_interior=True,
    colorbar_label: str = "",
):
    """
    Plot example prediction and grond truth.

    Each has shape (N_grid,)

    """
    if vrange is None:
        vmin = float(min(da_prediction.min(), da_target.min()))
        vmax = float(max(da_prediction.max(), da_target.max()))
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
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
        fig.suptitle(title, size=_TITLE_SIZE)

    cbar = fig.colorbar(
        axes[0].collections[0],
        ax=axes,
        orientation="horizontal",
        location="bottom",
        shrink=0.6,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=_TICK_SIZE)
    if colorbar_label:
        cbar.set_label(_tex_safe(colorbar_label), size=_LABEL_SIZE)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error: torch.Tensor,
    datastore: BaseRegularGridDatastore,
    title=None,
    vrange=None,
    boundary_alpha=0.7,
    crop_to_interior=True,
    colorbar_label: str = "",
):
    """Plot spatial error with projection-aware axes."""

    error_np = error.detach().cpu().numpy()

    if vrange is None:
        vmin = float(np.nanmin(error_np))
        vmax = float(np.nanmax(error_np))
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(6.5, 6),
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
        boundary_alpha=boundary_alpha,
        crop_to_interior=crop_to_interior,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        location="bottom",
        shrink=0.8,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=_TICK_SIZE)
    cbar.formatter.set_powerlimits((-3, 3))
    if colorbar_label:
        cbar.set_label(_tex_safe(colorbar_label), size=_LABEL_SIZE)

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)

    return fig
