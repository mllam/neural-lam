"""Visualization helpers for analysing Neural-LAM predictions and errors."""

# Standard library
import warnings
from typing import Optional

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore

# Font sizes shared across projection-aware plot functions.
_TITLE_SIZE = 13  # suptitle and per-axes titles
_LABEL_SIZE = 11  # axis / colorbar labels
_TICK_SIZE = 11  # tick labels

import os
import warnings
import torch
import pytorch_lightning as pl

# Annotations become unreadable when cells are smaller than this (in points)
# or when the total number of cells exceeds a readable count.
_MIN_CELL_SIZE_FOR_ANNOTATIONS = 18
_MAX_CELLS_FOR_ANNOTATIONS = 800
_HEATMAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "error_heatmap_white_red",
    ["#ffffff", "#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"],
)


def _tex_safe(s: str) -> str:
    """
    Escape TeX special characters in s if TeX rendering is active.

    Needed because % is a TeX comment character; without escaping it would
    silently truncate any text that follows it (e.g. the title for r2m (%)).

    Parameters
    ----------
    s : str
        The string to escape.

    Returns
    -------
    str
        The escaped string.
    """
    if plt.rcParams.get("text.usetex", False):
        s = s.replace("%", r"\%")
    return s


def _compute_heatmap_layout(n_rows: int, n_cols: int) -> dict[str, float]:
    """
    Choose figure and font sizes from the heatmap dimensions.

    Parameters
    ----------
    n_rows : int
        Number of rows in the heatmap.
    n_cols : int
        Number of columns in the heatmap.

    Returns
    -------
    dict[str, float]
        Dictionary containing figure width, figure height, font sizes, etc.
    """
    max_dim = max(n_rows, n_cols)

    # Size the figure so each cell gets ~0.8 x 0.5 inches; floor at 8 x 4.5.
    fig_width = float(max(4.5 + 0.8 * n_cols, 8.0))
    fig_height = float(max(2.5 + 0.5 * n_rows, 4.5))

    # Approximate cell size in points (72 pt/inch) to decide whether
    # in-cell annotations will be legible.
    cell_w_pt = (fig_width / max(n_cols, 1)) * 72
    cell_h_pt = (fig_height / max(n_rows, 1)) * 72
    show_annotations = (
        n_rows * n_cols <= _MAX_CELLS_FOR_ANNOTATIONS
        and min(cell_w_pt, cell_h_pt) >= _MIN_CELL_SIZE_FOR_ANNOTATIONS
    )

    return {
        "fig_width": fig_width,
        "fig_height": fig_height,
        "tick_label_size": float(np.clip(15.0 - 0.18 * max_dim, 9.0, 14.0)),
        "annotation_size": float(np.clip(13.0 - 0.22 * max_dim, 5.0, 12.0)),
        "title_size": float(np.clip(16.0 - 0.15 * max_dim, 9.0, 15.0)),
        "x_tick_rotation": 45.0 if n_cols > 12 else 0.0,
        "show_annotations": show_annotations,
    }


def _get_heatmap_var_labels(datastore: BaseRegularGridDatastore) -> list[str]:
    """
    Build state-variable labels from datastore metadata.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        The datastore containing metadata about the grid.

    Returns
    -------
    list[str]
        List of formatted variable labels.
    """
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    return [
        _tex_safe(f"{name} ({unit})" if unit else name)
        for name, unit in zip(var_names, var_units)
    ]


def _to_heatmap_matrix(values) -> np.ndarray:
    """
    Convert heatmap inputs to a ``(num_state_vars, pred_steps)`` matrix.

    A single-step tensor may arrive as one-dimensional ``(num_state_vars,)``,
    especially in single-GPU or focused metric logging paths. In that case we
    first treat it as one row of ``(pred_steps=1, num_state_vars)`` before
    transposing.

    Parameters
    ----------
    values : array-like
        The input values to convert.
        Shape ``(num_state_vars,)`` or ``(pred_steps, num_state_vars)``.

    Returns
    -------
    np.ndarray
        The converted heatmap matrix with shape
        ``(num_state_vars, pred_steps)``.
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    return values.T


def _get_feature_scale(
    ds_stats: xr.Dataset, var_name: str, n_vars: int
) -> np.ndarray | None:
    """
    Extract a 1D per-feature scale, averaging over any extra dims.

    Parameters
    ----------
    ds_stats : xr.Dataset
        The standardization statistics dataset.
    var_name : str
        The name of the variable to extract scale for.
    n_vars : int
        The number of variables expected.

    Returns
    -------
    np.ndarray or None
        The extracted scale as a 1D array, or None if unavailable.
    """
    if var_name not in ds_stats:
        return None

    da_scale = ds_stats[var_name]
    feature_dim = "state_feature"
    if feature_dim not in da_scale.dims:
        return None

    reduce_dims = [dim for dim in da_scale.dims if dim != feature_dim]
    if reduce_dims:
        da_scale = da_scale.mean(dim=reduce_dims)

    scale = np.asarray(da_scale.values, dtype=float).reshape(-1)
    if scale.size < n_vars:
        return None

    return scale[:n_vars]


def _get_heatmap_color_values(
    errors_np: np.ndarray,
    datastore: BaseRegularGridDatastore,
    normalization: str,
) -> tuple[np.ndarray, str, matplotlib.colors.Colormap]:
    """
    Normalize heatmap colors according to `normalization`.

    Returns a 3-tuple: (color_values, colorbar_label, cmap).
    The returned array drives the colormap only; the numeric annotations in the
    heatmap remain the original (physical-unit) values passed in `errors_np`.

    Both modes fall back to per-variable max normalization when their required
    stat is unavailable, appending "[fallback]" to the colorbar label.

    Parameters
    ----------
    errors_np : np.ndarray
        The error values to normalize.
    datastore : BaseRegularGridDatastore
        The datastore containing standardization stats.
    normalization : str
        The normalization mode to use ('state_std' or 'diff_std').

    Returns
    -------
    tuple[np.ndarray, str, matplotlib.colors.Colormap]
        A 3-tuple containing:
        - color_values: The normalized values for the colormap.
        - colorbar_label: The label for the colorbar.
        - cmap: The colormap to use.

    Raises
    ------
    ValueError
        If ``normalization`` is not one of ``'state_std'`` or ``'diff_std'``.
    """

    def _per_var_fallback():
        """
        Normalize errors by per-variable maximum value.

        Returns
        -------
        tuple[np.ndarray, str, matplotlib.colors.Colormap]
            Normalized errors, fallback label, and colormap.
        """
        max_err = errors_np.max(axis=1, keepdims=True)
        safe = np.where(max_err > np.finfo(float).eps, max_err, 1.0)
        return (
            errors_np / safe,
            "Per-variable scale (relative to max error) [fallback]",
            _HEATMAP_CMAP,
        )

    try:
        ds_stats = datastore.get_standardization_dataarray(category="state")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        warnings.warn(
            f"Could not load standardization stats ({exc}); "
            "falling back to per-variable scale.",
            UserWarning,
            stacklevel=3,
        )
        return _per_var_fallback()

    n_vars = errors_np.shape[0]

    if normalization == "state_std":
        state_std = _get_feature_scale(ds_stats, "state_std", n_vars)
        if state_std is None:
            warnings.warn(
                "state_std unavailable; falling back to per-variable scale.",
                UserWarning,
                stacklevel=3,
            )
            return _per_var_fallback()
        safe_std = np.where(
            np.isfinite(state_std) & (state_std > np.finfo(float).eps),
            state_std,
            1.0,
        )
        return errors_np / safe_std[:, None], "Error / state_std", _HEATMAP_CMAP

    if normalization == "diff_std":
        diff_std_std = _get_feature_scale(
            ds_stats, "state_diff_std_standardized", n_vars
        )
        if diff_std_std is None:
            warnings.warn(
                "state_diff_std_standardized unavailable; "
                "falling back to per-variable scale.",
                UserWarning,
                stacklevel=3,
            )
            return _per_var_fallback()
        state_std = _get_feature_scale(ds_stats, "state_std", n_vars)
        if state_std is None:
            warnings.warn(
                "state_std unavailable (needed to recover physical diff_std); "
                "falling back to per-variable scale.",
                UserWarning,
                stacklevel=3,
            )
            return _per_var_fallback()
        scale = state_std * diff_std_std  # physical diff_std
        safe = np.where(
            np.isfinite(scale) & (np.abs(scale) > np.finfo(float).eps),
            scale,
            1.0,
        )
        return (
            errors_np / safe[:, None],
            "Error / physical diff_std",
            _HEATMAP_CMAP,
        )

    raise ValueError(
        f"Unknown normalization {normalization!r}; "
        "expected 'state_std' or 'diff_std'."
    )


def _get_annotation_text_color(
    value: float, image: matplotlib.image.AxesImage
) -> str:
    """
    Choose a readable annotation color from the rendered background.

    Parameters
    ----------
    value : float
        The numeric value at the cell.
    image : matplotlib.image.AxesImage
        The rendered image object to determine background color.

    Returns
    -------
    str
        'white' or 'black' depending on the background luminance.
    """
    if not np.isfinite(value):
        return "black"

    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.5 else "black"


def plot_on_axis(
    ax: matplotlib.axes.Axes,
    da: xr.DataArray,
    datastore: BaseRegularGridDatastore,
    vmin: float | None = None,
    vmax: float | None = None,
    ax_title: str | None = None,
    cmap: str | matplotlib.colors.Colormap = "plasma",
    boundary_alpha: float | None = None,
    crop_to_interior: bool = False,
) -> matplotlib.collections.QuadMesh:
    """
    Plot weather state on a projection-aware axis using datastore metadata.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on. Should have a cartopy projection.
    da : xarray.DataArray
        The data to plot. Should have shape (num_grid_nodes,).
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


# rc_context applies NeurIPS font/text settings; figure size is overridden
# by the explicit figsize= below but font family and usetex stay in effect.
@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_heatmap(
    errors: torch.Tensor,
    datastore: BaseRegularGridDatastore,
    title: Optional[str] = None,
    normalization: str = "state_std",
) -> matplotlib.figure.Figure:
    """
    Plot a heatmap of errors across variables and prediction horizons.

    Parameters
    ----------
    errors : torch.Tensor
        Shape ``(pred_steps, num_state_vars)``. Per-step, per-variable
        errors. These values are used for the numeric annotations in each
        cell.
    datastore : BaseRegularGridDatastore
        Datastore providing variable names, units, and step length.
    title : str, optional
        Title for the axes.
    normalization : {"state_std", "diff_std"}, default "state_std"
        Color scaling mode. "state_std" divides by climatological std;
        "diff_std" divides by the typical one-step change magnitude. Both fall
        back to per-variable max error when the required stats are missing.

    Returns
    -------
    matplotlib.figure.Figure
        The completed heatmap figure.

    Notes
    -----
    Color scaling is controlled by `normalization`; see
    `_get_heatmap_color_values` for the full fallback logic. When stats are
    unavailable the colorbar label includes "[fallback]".
    """
    errors_np = _to_heatmap_matrix(errors)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    time_step_int, time_step_unit = utils.get_integer_time(step_length)
    layout = _compute_heatmap_layout(n_rows=d_f, n_cols=pred_steps)
    color_values_np, colorbar_label, heatmap_cmap = _get_heatmap_color_values(
        errors_np, datastore, normalization
    )
    finite_color_values = color_values_np[np.isfinite(color_values_np)]
    if finite_color_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite_color_values.min())
        vmax = float(finite_color_values.max())
        vmin = min(0.0, vmin)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    fig, ax = plt.subplots(
        figsize=(layout["fig_width"], layout["fig_height"]),
        constrained_layout=True,
    )

    im = ax.imshow(
        color_values_np,
        cmap=heatmap_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(_tex_safe(colorbar_label), size=layout["tick_label_size"])
    cbar.ax.tick_params(labelsize=layout["tick_label_size"])
    cbar.ax.yaxis.get_offset_text().set_fontsize(layout["tick_label_size"])

    if layout["show_annotations"]:
        for (j, i), error in np.ndenumerate(errors_np):
            if np.isfinite(error):
                formatted_error = (
                    f"{error:.3g}" if abs(error) < 1.0e4 else f"{error:.2E}"
                )
            else:
                formatted_error = str(error)
            text_color = _get_annotation_text_color(color_values_np[j, i], im)
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
        ha="right" if layout["x_tick_rotation"] > 0 else "center",
    )
    ax.set_xlabel(
        f"Lead time ({time_step_unit[0]})", size=layout["tick_label_size"]
    )

    ax.set_yticks(np.arange(d_f))
    ax.set_yticklabels(
        _get_heatmap_var_labels(datastore=datastore),
        size=layout["tick_label_size"],
    )

    if title:
        ax.set_title(title, size=layout["title_size"])

    return fig


def plot_error_map(
    errors: torch.Tensor,
    datastore: BaseRegularGridDatastore,
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Deprecated: use :func:`plot_error_heatmap` instead.

    Parameters
    ----------
    errors : torch.Tensor
        The error values to plot.
    datastore : BaseRegularGridDatastore
        The datastore containing grid metadata.
    title : str, optional
        The title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The completed heatmap figure.
    """
    warnings.warn(
        "plot_error_map is deprecated, use plot_error_heatmap instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_error_heatmap(errors, datastore=datastore, title=title)


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore,
    da_prediction: xr.DataArray,
    da_target: xr.DataArray,
    title: Optional[str] = None,
    vrange: Optional[tuple[float, float]] = None,
    boundary_alpha: float = 0.7,
    crop_to_interior: bool = True,
    colorbar_label: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot an example prediction alongside the ground truth.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
       Datastore providing grid metadata and projection.
    da_prediction : xarray.DataArray
        Shape ``(num_grid_nodes,)``. Predicted field values.
    da_target : xarray.DataArray
        Shape ``(num_grid_nodes,)``. Ground-truth field values.
    title : str, optional
        Overall figure title.
    vrange : tuple of (float, float), optional
        ``(vmin, vmax)`` for the shared colour scale. Inferred from data
        if not given.
    boundary_alpha : float, optional
        Alpha transparency for the boundary overlay (default 0.7).
    crop_to_interior : bool, optional
        If True, crop the axes to the interior region (default True).
    colorbar_label : str, optional
        Label for the shared colorbar.

    Returns
    -------
    matplotlib.figure.Figure
        The completed two-panel prediction figure.
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
    title: Optional[str] = None,
    vrange: Optional[tuple[float, float]] = None,
    boundary_alpha: float = 0.7,
    crop_to_interior: bool = True,
    colorbar_label: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot a spatially resolved error map on a projection-aware axis.

    Parameters
    ----------
    error : torch.Tensor
        Error magnitudes on the flattened grid.
        * **Shape**: ``(num_grid_nodes,)``
    datastore : BaseRegularGridDatastore
        Datastore providing coordinate metadata and boundary masks.
    title : str or None, optional
        Optional figure title.
    vrange : tuple[float, float] or None, optional
        Explicit value range ``(vmin, vmax)`` for the color scale.
    boundary_alpha : float, optional
        Alpha transparency for the boundary overlay (default 0.7).
    crop_to_interior : bool, optional
        If True, crop the axes to the interior region (default True).
    colorbar_label : str, optional
        Label for the colorbar.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle containing the plotted map.
    """
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

        # Call directly on datastore now!
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
