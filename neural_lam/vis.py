# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors
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


def _compute_heatmap_layout(n_rows: int, n_cols: int) -> dict[str, float]:
    """Choose figure and font sizes from the heatmap dimensions."""
    max_dim = max(n_rows, n_cols)

    return {
        "fig_width": float(np.clip(4.5 + 0.8 * n_cols, 8.0, 18.0)),
        "fig_height": float(np.clip(2.5 + 0.5 * n_rows, 4.5, 18.0)),
        "tick_label_size": float(np.clip(15.0 - 0.18 * max_dim, 7.0, 14.0)),
        "annotation_size": float(np.clip(13.0 - 0.22 * max_dim, 5.0, 12.0)),
        "title_size": float(np.clip(16.0 - 0.15 * max_dim, 9.0, 15.0)),
        "x_tick_rotation": 45.0 if n_cols > 12 else 0.0,
    }


def _get_heatmap_var_labels(
    datastore: BaseRegularGridDatastore, n_vars: int
) -> list[str]:
    """Build state-variable labels, padding defensively if metadata is short."""
    var_names = list(datastore.get_vars_names(category="state"))
    var_units = list(datastore.get_vars_units(category="state"))

    if len(var_names) < n_vars:
        var_names.extend(
            [f"state_feature_{i}" for i in range(len(var_names), n_vars)]
        )
    if len(var_units) < n_vars:
        var_units.extend([""] * (n_vars - len(var_units)))

    labels = []
    for name, unit in zip(var_names[:n_vars], var_units[:n_vars]):
        labels.append(f"{name} ({unit})" if unit else name)

    return labels


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_heatmap(
    errors, datastore: BaseRegularGridDatastore, title=None
):
    """
    Plot a heatmap of errors for state variables across forecast lead times.

    Parameters
    ----------
    errors : torch.Tensor
        Error values with shape `(pred_steps, d_f)`.
    datastore : BaseRegularGridDatastore
        Datastore providing step length and variable metadata.
    title : str, optional
        Optional title for the figure.
    """
    errors_np = errors.detach().cpu().numpy().T  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    time_step_int, time_step_unit = utils.get_integer_time(step_length)
    layout = _compute_heatmap_layout(n_rows=d_f, n_cols=pred_steps)

    finite_errors = errors_np[np.isfinite(errors_np)]
    if finite_errors.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite_errors.min())
        vmax = float(finite_errors.max())
        if vmin >= 0.0:
            vmin = 0.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    fig, ax = plt.subplots(
        figsize=(layout["fig_width"], layout["fig_height"]),
        constrained_layout=True,
    )

    im = ax.imshow(
        errors_np,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=layout["tick_label_size"])
    cbar.ax.yaxis.get_offset_text().set_fontsize(layout["tick_label_size"])

    for (j, i), error in np.ndenumerate(errors_np):
        formatted_error = f"{error:.3g}" if abs(error) < 1.0e4 else f"{error:.2E}"
        text_color = "white" if im.norm(error) < 0.45 else "black"
        ax.text(
            i,
            j,
            formatted_error,
            ha="center",
            va="center",
            usetex=False,
            fontsize=layout["annotation_size"],
            color=text_color,
        )

    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1
    pred_hor_h = time_step_int * pred_hor_i
    ax.set_xticklabels(
        pred_hor_h,
        size=layout["tick_label_size"],
        rotation=layout["x_tick_rotation"],
        ha="right" if layout["x_tick_rotation"] else "center",
    )
    ax.set_xlabel(
        f"Lead time ({time_step_unit[0]})", size=layout["tick_label_size"]
    )

    ax.set_yticks(np.arange(d_f))
    ax.set_yticklabels(
        _get_heatmap_var_labels(datastore=datastore, n_vars=d_f),
        size=layout["tick_label_size"],
    )

    if title:
        ax.set_title(title, size=layout["title_size"])

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

    mesh = plot_on_axis(
        ax=ax,
        da=xr.DataArray(error_np),
        datastore=datastore,
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
