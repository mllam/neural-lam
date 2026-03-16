# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Standard library
import warnings

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore

# Annotations become unreadable when cells are smaller than this (in points)
# or when the total number of cells exceeds a readable count.
_MIN_CELL_SIZE_FOR_ANNOTATIONS = 18
_MAX_CELLS_FOR_ANNOTATIONS = 800
_HEATMAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "error_heatmap_white_red",
    ["#ffffff", "#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"],
)


def _compute_heatmap_layout(n_rows: int, n_cols: int) -> dict[str, float]:
    """Choose figure and font sizes from the heatmap dimensions.

    Scaling coefficients were empirically tuned for readability on grids
    ranging from ~5 to ~50 variables/lead-times.  The figure grows
    proportionally so that cells never shrink below a readable size.
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
        "tick_label_size": float(np.clip(15.0 - 0.18 * max_dim, 7.0, 14.0)),
        "annotation_size": float(np.clip(13.0 - 0.22 * max_dim, 5.0, 12.0)),
        "title_size": float(np.clip(16.0 - 0.15 * max_dim, 9.0, 15.0)),
        "x_tick_rotation": 45.0 if n_cols > 12 else 0.0,
        "show_annotations": show_annotations,
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


def _to_heatmap_matrix(values) -> np.ndarray:
    """Convert `(pred_steps, d_f)` values to a `(d_f, pred_steps)` matrix."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).T


def _get_feature_scale(
    ds_stats: xr.Dataset, var_name: str, n_vars: int
) -> np.ndarray | None:
    """Extract a 1D per-feature scale, averaging over any extra dims."""
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
    errors_np: np.ndarray, datastore: BaseRegularGridDatastore
) -> tuple[np.ndarray, str]:
    """Normalize heatmap colors to a cross-variable relative scale.

    The background colors should compare relative error magnitudes across
    variables, while the text annotations keep the original values/units.
    When one-step-difference statistics are available, normalize by the
    corresponding physical-scale diff std so colors show error relative to
    typical one-step changes of each variable.
    """
    try:
        ds_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
    except (AttributeError, KeyError, ValueError, TypeError):
        return errors_np, "Relative scale"

    n_vars = errors_np.shape[0]
    state_std = _get_feature_scale(ds_state_stats, "state_std", n_vars)
    if state_std is None:
        return errors_np, "Relative scale"

    scale = state_std
    colorbar_label = "Relative scale (state stds)"

    state_diff_std_standardized = _get_feature_scale(
        ds_state_stats, "state_diff_std_standardized", n_vars
    )
    if state_diff_std_standardized is not None:
        scale = scale * state_diff_std_standardized
        colorbar_label = "Relative scale (1-step diff stds)"

    safe_scale = np.where(
        np.isfinite(scale) & (np.abs(scale) > np.finfo(float).eps),
        scale,
        1.0,
    )
    return errors_np / safe_scale[:, None], colorbar_label


def _get_annotation_text_color(
    value: float, image: matplotlib.image.AxesImage
) -> str:
    """Choose a readable annotation color from the rendered background."""
    if not np.isfinite(value):
        return "black"

    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.5 else "black"


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_heatmap(
    errors,
    datastore: BaseRegularGridDatastore,
    title=None,
    color_values=None,
    colorbar_label=None,
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
    color_values : torch.Tensor, optional
        Optional values used only for the background colors. If omitted,
        colors are normalized from ``errors`` using datastore state-variable
        standardization statistics.
    colorbar_label : str, optional
        Optional label for the colorbar. If omitted, an automatic label is
        chosen based on the color normalization used.
    """
    errors_np = _to_heatmap_matrix(errors)  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    time_step_int, time_step_unit = utils.get_integer_time(step_length)
    layout = _compute_heatmap_layout(n_rows=d_f, n_cols=pred_steps)

    if color_values is None:
        color_values_np, default_colorbar_label = _get_heatmap_color_values(
            errors_np, datastore
        )
    else:
        color_values_np = _to_heatmap_matrix(color_values)
        default_colorbar_label = "Relative scale"
        if color_values_np.shape != errors_np.shape:
            raise ValueError(
                "color_values must have the same shape as errors: "
                f"got {color_values_np.T.shape} and {errors_np.T.shape}"
            )

    if colorbar_label is None:
        colorbar_label = default_colorbar_label

    finite_color_values = color_values_np[np.isfinite(color_values_np)]
    if finite_color_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite_color_values.min())
        vmax = float(finite_color_values.max())
        if vmin >= 0.0:
            vmin = 0.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    fig, ax = plt.subplots(
        figsize=(layout["fig_width"], layout["fig_height"]),
        constrained_layout=True,
    )

    im = ax.imshow(
        color_values_np,
        cmap=_HEATMAP_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label, size=layout["tick_label_size"])
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
        _get_heatmap_var_labels(datastore=datastore, n_vars=d_f),
        size=layout["tick_label_size"],
    )

    if title:
        ax.set_title(title, size=layout["title_size"])

    return fig


def plot_error_map(errors, datastore: BaseRegularGridDatastore, title=None):
    """Deprecated: use :func:`plot_error_heatmap` instead."""
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
