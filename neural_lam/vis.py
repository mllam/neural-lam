# Standard library
import io

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


def get_var_cmap(var_name: str) -> str:
    """Return an appropriate matplotlib colormap for a given variable name.

    Implements fix 1.3: diverging maps for anomaly/error fields, YlOrRd for
    uncertainty/std fields, viridis for everything else.
    Callers can use this to pick a cmap before calling plot_on_axis or
    plot_prediction rather than always passing "plasma".

    Examples
    --------
    >>> get_var_cmap("t_2m_anom")   # → "RdBu_r"
    >>> get_var_cmap("ens_std")     # → "YlOrRd"
    >>> get_var_cmap("u_10m")       # → "viridis"
    """
    low = var_name.lower()
    # Anomaly / difference / bias fields  → diverging (sign matters)
    if any(
        k in low for k in ("anom", "anomaly", "diff", "bias", "error", "err")
    ):
        return "RdBu_r"
    # Uncertainty / spread / std fields   → sequential warm (intensity matters)
    if any(k in low for k in ("std", "spread", "unc", "uncertainty", "crps")):
        return "YlOrRd"
    # Everything else                     → perceptually uniform sequential
    return "viridis"


def _to_numpy(data) -> np.ndarray:
    """Convert tensor, DataArray, or array to a detached 1-D numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, xr.DataArray):
        return data.values
    return np.asarray(data)


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
    da : xarray.DataArray or numpy.ndarray or torch.Tensor
        The data to plot. Should be flat with N_grid elements.
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

    # Normalise to DataArray so downstream code can call .values uniformly
    if isinstance(da, (np.ndarray, torch.Tensor)):
        da = xr.DataArray(_to_numpy(da))

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
def plot_error_map(errors, datastore: BaseRegularGridDatastore, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """

    # Ensure errors is 2D even for single-step/single-GPU runs
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)

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
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1
    pred_hor_h = time_step_int * pred_hor_i
    ax.set_xticklabels(pred_hor_h, size=_TICK_SIZE)
    ax.set_xlabel(f"Lead time ({time_step_unit[0]})", size=_LABEL_SIZE)

    ax.set_yticks(np.arange(d_f))
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    y_ticklabels = [
        _tex_safe(f"{name} ({unit})")
        for name, unit in zip(var_names, var_units)
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=_TICK_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map_absolute(
    errors, datastore: BaseRegularGridDatastore, title=None
):
    """Plot a heatmap of raw (unnormalized) metric values.

    Unlike plot_error_map, each row is NOT divided by its own maximum, so
    errors across variables can be compared directly on a shared colour scale.
    Pairs with plot_error_map to answer "which variable is hardest to predict?"

    errors: (pred_steps, d_f)
    """
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)

    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length
    time_step_int, time_step_unit = utils.get_integer_time(step_length)

    global_max = errors_np.max()
    global_max = global_max if global_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_np,
        cmap="OrRd",
        vmin=0,
        vmax=global_max,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    for (j, i), error in np.ndenumerate(errors_np):
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    ax.set_xticks(np.arange(pred_steps))
    pred_hor_h = time_step_int * (np.arange(pred_steps) + 1)
    ax.set_xticklabels(pred_hor_h, size=_TICK_SIZE)
    ax.set_xlabel(f"Lead time ({time_step_unit[0]})", size=_LABEL_SIZE)

    ax.set_yticks(np.arange(d_f))
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    y_ticklabels = [
        _tex_safe(f"{name} ({unit})")
        for name, unit in zip(var_names, var_units)
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=_TICK_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title(
            "Absolute metric values (shared colour scale)", size=_TITLE_SIZE
        )

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
    cmap: str = "viridis",
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
            cmap=cmap,
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


# ---------------------------------------------------------------------------
# Ported from prob_model_lam — ensemble and latent-space plots
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    samples,
    target,
    ens_mean,
    ens_std,
    datastore: BaseRegularGridDatastore,
    title=None,
    vrange=None,
    vrange_std=None,
    boundary_alpha=0.7,
    crop_to_interior=False,
):
    """Plot ensemble prediction: target, mean, std and up to 6 members.

    Parameters
    ----------
    samples : (S, N_grid) tensor or array
    target : (N_grid,) tensor or array
    ens_mean : (N_grid,) tensor or array
    ens_std : (N_grid,) tensor or array
    datastore : BaseRegularGridDatastore
    vrange : (vmin, vmax) for value panels; auto if None
    vrange_std : (vmin, vmax) for the std panel; auto if None.
        Pass a fixed range across lead times to make spread growth visible.
    """
    samples_np = _to_numpy(samples)  # (S, N_grid)
    target_np = _to_numpy(target)
    ens_mean_np = _to_numpy(ens_mean)
    ens_std_np = _to_numpy(ens_std)

    if vrange is None:
        vmin = float(min(samples_np.min(), target_np.min()))
        vmax = float(max(samples_np.max(), target_np.max()))
    else:
        vmin, vmax = vrange

    if vrange_std is None:
        std_vmin = 0.0
        std_vmax = float(ens_std_np.max())
    else:
        std_vmin, std_vmax = vrange_std

    n_members = min(samples_np.shape[0], 6)
    n_panels = 3 + n_members  # target, mean, std + members
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        subplot_kw={"projection": datastore.coords_projection},
    )
    axes_flat = np.array(axes).flatten()

    # Target
    gt_mesh = plot_on_axis(
        axes_flat[0],
        target_np,
        datastore=datastore,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ground Truth",
        cmap="viridis",
        boundary_alpha=boundary_alpha,
        crop_to_interior=crop_to_interior,
    )
    # Ensemble mean
    plot_on_axis(
        axes_flat[1],
        ens_mean_np,
        datastore=datastore,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ens. Mean",
        cmap="viridis",
        boundary_alpha=boundary_alpha,
        crop_to_interior=crop_to_interior,
    )
    # Ensemble std — use YlOrRd so uncertainty reads as darker/warmer
    std_mesh = plot_on_axis(
        axes_flat[2],
        ens_std_np,
        datastore=datastore,
        vmin=std_vmin,
        vmax=std_vmax,
        ax_title="Ens. Std.",
        cmap="YlOrRd",
        boundary_alpha=boundary_alpha,
        crop_to_interior=crop_to_interior,
    )

    # Individual members
    for member_i in range(n_members):
        plot_on_axis(
            axes_flat[3 + member_i],
            samples_np[member_i],
            datastore=datastore,
            vmin=vmin,
            vmax=vmax,
            ax_title=f"Member {member_i + 1}",
            cmap="viridis",
            boundary_alpha=boundary_alpha,
            crop_to_interior=crop_to_interior,
        )

    # Turn off unused axes
    for ax in axes_flat[3 + n_members :]:
        ax.axis("off")

    # Colorbars
    values_axes = list(axes_flat[:2]) + list(axes_flat[3 : 3 + n_members])
    values_cbar = fig.colorbar(
        gt_mesh,
        ax=values_axes,
        orientation="horizontal",
        location="bottom",
        shrink=0.8,
        pad=0.02,
    )
    values_cbar.ax.tick_params(labelsize=_TICK_SIZE)

    std_cbar = fig.colorbar(
        std_mesh,
        ax=axes_flat[2],
        orientation="horizontal",
        location="bottom",
        shrink=0.8,
        pad=0.02,
    )
    std_cbar.ax.tick_params(labelsize=_TICK_SIZE)

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """Plot samples of latent variable from prior and variational distribution.

    prior_samples: (samples, N_mesh, d_latent)
    vi_samples: (samples, N_mesh, d_latent)
    """
    num_samples, num_mesh_nodes, latent_dim = prior_samples.shape
    plot_dims = min(latent_dim, 3)
    img_side_size = int(np.sqrt(num_mesh_nodes))
    assert img_side_size**2 == num_mesh_nodes, (
        "Number of mesh nodes is not a square number, "
        "cannot plot latent samples as images"
    )

    vmin = min(
        _to_numpy(vals[..., :plot_dims]).min()
        for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        _to_numpy(vals[..., :plot_dims]).max()
        for vals in (prior_samples, vi_samples)
    )

    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):
        for dim_i in range(plot_dims):
            prior_reshaped = _to_numpy(prior_sample[:, dim_i]).reshape(
                img_side_size, img_side_size
            )
            vi_reshaped = _to_numpy(vi_sample[:, dim_i]).reshape(
                img_side_size, img_side_size
            )
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                prior_ax.set_title(f"d{dim_i} (prior)", size=_TITLE_SIZE)
                vi_ax.set_title(f"d{dim_i} (vi)", size=_TITLE_SIZE)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=_TICK_SIZE)

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)

    return fig


# ---------------------------------------------------------------------------
# Part 2.1 — Spread–skill diagram
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spread_skill(spread, rmse, var_names, title=None):
    """Spread–skill diagram: scatter of ensemble spread vs RMSE.

    A well-calibrated ensemble lies along the 1:1 diagonal.
    Points below the diagonal indicate over-confidence (spread < error);
    points above indicate under-confidence.

    Parameters
    ----------
    spread : (pred_steps, d_f) array-like — ensemble spread per step/variable
    rmse : (pred_steps, d_f) array-like — RMSE of ensemble mean
    var_names : list[str] — variable names for subplot titles
    """
    spread_np = _to_numpy(spread)  # (pred_steps, d_f)
    rmse_np = _to_numpy(rmse)
    d_f = spread_np.shape[1]

    ncols = min(d_f, 4)
    nrows = (d_f + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if d_f > 1 else [axes]

    global_max = max(spread_np.max(), rmse_np.max())

    for var_i, (ax, var_name) in enumerate(zip(axes_flat, var_names)):
        s = spread_np[:, var_i]
        r = rmse_np[:, var_i]

        ax.scatter(s, r, alpha=0.7, s=20, color="steelblue")
        # 1:1 reference line
        ref = [0, global_max]
        ax.plot(ref, ref, "k--", linewidth=1, label="1:1")

        ax.set_xlim(0, global_max * 1.05)
        ax.set_ylim(0, global_max * 1.05)
        ax.set_xlabel("Ensemble spread", size=_LABEL_SIZE)
        ax.set_ylabel("RMSE", size=_LABEL_SIZE)
        ax.set_title(_tex_safe(var_name), size=_TITLE_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)
        ax.legend(fontsize=_TICK_SIZE)

    for ax in axes_flat[d_f:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)
    else:
        fig.suptitle("Spread–Skill Diagram", size=_TITLE_SIZE)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.2 — Rank histogram (Talagrand diagram)
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_rank_histogram(ranks, n_members, var_names, title=None):
    """Rank (Talagrand) histogram for ensemble calibration.

    A flat histogram indicates a well-calibrated ensemble.
    A U-shape indicates under-dispersion; a dome indicates over-dispersion.

    Parameters
    ----------
    ranks : (d_f, n_obs) array-like — rank of observed truth among ensemble
        members for each variable. Values should be in [0, n_members].
    n_members : int — number of ensemble members
    var_names : list[str]
    """
    ranks_np = _to_numpy(ranks)  # (d_f, n_obs)
    if ranks_np.ndim == 1:
        ranks_np = ranks_np[np.newaxis, :]  # treat as single variable
    d_f = ranks_np.shape[0]
    n_bins = n_members + 1

    ncols = min(d_f, 4)
    nrows = (d_f + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if d_f > 1 else [axes]

    for var_i, (ax, var_name) in enumerate(zip(axes_flat, var_names)):
        counts, _ = np.histogram(
            ranks_np[var_i], bins=n_bins, range=(0, n_members + 1)
        )
        n_obs = len(ranks_np[var_i])
        flat_level = n_obs / n_bins

        ax.bar(
            np.arange(n_bins),
            counts,
            width=0.9,
            color="steelblue",
            alpha=0.7,
        )
        ax.axhline(
            flat_level,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Flat (perfect)",
        )

        ax.set_xlabel("Rank", size=_LABEL_SIZE)
        ax.set_ylabel("Count", size=_LABEL_SIZE)
        ax.set_title(_tex_safe(var_name), size=_TITLE_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)
        ax.legend(fontsize=_TICK_SIZE)

    for ax in axes_flat[d_f:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)
    else:
        fig.suptitle("Rank Histogram (Talagrand Diagram)", size=_TITLE_SIZE)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.3 — Reliability diagram
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_reliability_diagram(
    forecast_probs, observed_freqs, thresholds, var_names, title=None
):
    """Reliability diagram for probabilistic calibration.

    A model with perfect reliability produces points along the 45° diagonal.
    Points below the diagonal indicate over-confidence; above the diagonal
    indicates under-confidence.

    Parameters
    ----------
    forecast_probs : (d_f, n_thresholds) array-like — predicted probabilities
    observed_freqs : (d_f, n_thresholds) array-like — observed event
        frequencies
    thresholds : (n_thresholds,) array-like — probability threshold levels in
        [0, 1]
    var_names : list[str]
    """
    fp = _to_numpy(forecast_probs)  # (d_f, n_thresholds)
    of = _to_numpy(observed_freqs)

    if fp.ndim == 1:
        fp = fp[np.newaxis, :]
        of = of[np.newaxis, :]
    d_f = fp.shape[0]

    ncols = min(d_f, 4)
    nrows = (d_f + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if d_f > 1 else [axes]

    for var_i, (ax, var_name) in enumerate(zip(axes_flat, var_names)):
        ax.plot(
            fp[var_i],
            of[var_i],
            "o-",
            color="steelblue",
            alpha=0.8,
            label="Model",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        # Shade gap between perfect calibration (y=x) and model curve.
        # Both x and y use fp[var_i] so the shading is on the same axis scale
        # as the scatter.
        ax.fill_between(
            fp[var_i], fp[var_i], of[var_i], alpha=0.1, color="steelblue"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability", size=_LABEL_SIZE)
        ax.set_ylabel("Observed frequency", size=_LABEL_SIZE)
        ax.set_title(_tex_safe(var_name), size=_TITLE_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)
        ax.legend(fontsize=_TICK_SIZE)

    for ax in axes_flat[d_f:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)
    else:
        fig.suptitle("Reliability Diagram", size=_TITLE_SIZE)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.4 — CRPS vs. lead time line plot
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_crps_leadtime(
    crps_tensor, datastore: BaseRegularGridDatastore, title=None
):
    """Line plot of CRPS vs lead time, one line per variable.

    Complements plot_error_map by making temporal trends immediately visible.
    A sudden jump at a particular lead time signals a structural rollout issue.

    Parameters
    ----------
    crps_tensor : (pred_steps, d_f) tensor — CRPS in physical units
    datastore : BaseRegularGridDatastore
    """
    if isinstance(crps_tensor, torch.Tensor):
        crps_np = crps_tensor.detach().cpu().numpy()
    else:
        crps_np = np.asarray(crps_tensor)

    pred_steps, d_f = crps_np.shape
    step_length = datastore.step_length
    time_step_int, time_step_unit = utils.get_integer_time(step_length)
    lead_times = time_step_int * (np.arange(pred_steps) + 1)
    var_names = datastore.get_vars_names(category="state")

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("tab20")
    for var_i, var_name in enumerate(var_names):
        ax.plot(
            lead_times,
            crps_np[:, var_i],
            label=_tex_safe(var_name),
            color=cmap(var_i % 20),
            linewidth=1.5,
        )

    ax.set_xlabel(f"Lead time ({time_step_unit[0]})", size=_LABEL_SIZE)
    ax.set_ylabel("CRPS (physical units)", size=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(
        fontsize=max(6, _TICK_SIZE - 2),
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title("CRPS vs. Lead Time", size=_TITLE_SIZE)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.5 — Spaghetti / contour-bundle plot
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spaghetti(
    samples,
    contour_level,
    var_name,
    datastore: BaseRegularGridDatastore,
    title=None,
    alpha_per_member: float = 0.4,
):
    """Spaghetti / contour-bundle plot for ensemble spatial coherence.

    Each ensemble member's contour at *contour_level* is drawn as a thin line.
    Tightly bundled lines → confident ensemble; splayed lines → large spread.

    Parameters
    ----------
    samples : (S, N_grid) array-like — ensemble members
    contour_level : float — geophysical contour value to draw
    var_name : str — variable name for title
    datastore : BaseRegularGridDatastore
    alpha_per_member : float — line transparency (lower = more readable
        when S is large)
    """
    samples_np = _to_numpy(samples)  # (S, N_grid)
    n_members = samples_np.shape[0]

    lats_lons = datastore.get_lat_lon("state")
    grid_shape = (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    lons = lats_lons[:, 0].reshape(grid_shape)
    lats = lats_lons[:, 1].reshape(grid_shape)

    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={"projection": datastore.coords_projection},
    )
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)

    cmap = plt.get_cmap("Blues")
    for member_i in range(n_members):
        member_grid = samples_np[member_i].reshape(grid_shape)
        try:
            ax.contour(
                lons,
                lats,
                member_grid,
                levels=[contour_level],
                colors=[cmap(0.5 + 0.4 * member_i / max(n_members - 1, 1))],
                linewidths=0.8,
                alpha=alpha_per_member,
                transform=ccrs.PlateCarree(),
            )
        except Exception:
            # Contour may fail if the level isn't present in this member
            pass

    ax.set_title(
        _tex_safe(
            f"{var_name} — spaghetti at {contour_level:.2g} "
            f"({n_members} members)"
        ),
        size=_TITLE_SIZE,
    )

    if title:
        fig.suptitle(title, size=_TITLE_SIZE)

    return fig


# ---------------------------------------------------------------------------
# Part 2.6 — Plume / fan diagram
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_plume(
    ensemble_timeseries,
    target_timeseries,
    step_length,
    location_name: str,
    var_name: str,
    title=None,
):
    """Plume / fan diagram: forecast time series with uncertainty bands.

    The observed truth is a solid black line. Ensemble spread is shown as
    shaded percentile bands (10–90 and 25–75) with the median as a line.

    Parameters
    ----------
    ensemble_timeseries : (S, T) array-like — ensemble members over time
    target_timeseries : (T,) array-like — observed truth
    step_length : timedelta or numeric hours — forecast step length
    location_name : str — name of the geographic point
    var_name : str — variable name
    """
    ens_np = _to_numpy(ensemble_timeseries)  # (S, T)
    target_np = _to_numpy(target_timeseries)  # (T,)
    n_steps = ens_np.shape[1]

    if hasattr(step_length, "total_seconds"):
        step_hours = step_length.total_seconds() / 3600
    else:
        step_hours = float(step_length)

    lead_times = np.arange(1, n_steps + 1) * step_hours

    p10 = np.percentile(ens_np, 10, axis=0)
    p25 = np.percentile(ens_np, 25, axis=0)
    p50 = np.percentile(ens_np, 50, axis=0)
    p75 = np.percentile(ens_np, 75, axis=0)
    p90 = np.percentile(ens_np, 90, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(
        lead_times,
        p10,
        p90,
        alpha=0.25,
        color="steelblue",
        label="10–90th pct.",
    )
    ax.fill_between(
        lead_times,
        p25,
        p75,
        alpha=0.45,
        color="steelblue",
        label="25–75th pct.",
    )
    ax.plot(lead_times, p50, color="steelblue", linewidth=1.5, label="Median")
    ax.plot(
        lead_times,
        target_np,
        color="black",
        linewidth=1.5,
        linestyle="-",
        label="Observation",
    )

    ax.set_xlabel("Lead time (h)", size=_LABEL_SIZE)
    ax.set_ylabel(_tex_safe(var_name), size=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_TICK_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title(
            _tex_safe(f"{var_name} — {location_name}"), size=_TITLE_SIZE
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.7 — Probability-of-exceedance map
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_exceedance_prob(
    samples,
    threshold: float,
    var_name: str,
    datastore: BaseRegularGridDatastore,
    title=None,
    boundary_alpha: float = 0.7,
    crop_to_interior: bool = False,
):
    """Probability-of-exceedance spatial map.

    Shows the fraction of ensemble members that exceed *threshold* at each
    grid point. Directly useful for frost warnings, extreme precipitation, etc.

    Parameters
    ----------
    samples : (S, N_grid) array-like — ensemble members
    threshold : float — exceedance threshold in physical units
    var_name : str
    datastore : BaseRegularGridDatastore
    """
    samples_np = _to_numpy(samples)  # (S, N_grid)
    exceed_prob = (samples_np > threshold).mean(axis=0)  # (N_grid,)

    fig, ax = plt.subplots(
        figsize=(7, 6),
        subplot_kw={"projection": datastore.coords_projection},
    )

    mesh = plot_on_axis(
        ax=ax,
        da=exceed_prob,
        datastore=datastore,
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
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
    cbar.set_label("Exceedance probability", size=_LABEL_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title(
            _tex_safe(f"P({var_name} > {threshold:.3g})"), size=_TITLE_SIZE
        )

    return fig


# ---------------------------------------------------------------------------
# Part 2.8 — Animated GIF from a sequence of figures
# ---------------------------------------------------------------------------


def save_forecast_animation(figs, filename: str, duration_ms: int = 700):
    """Save a list of matplotlib figures as an animated GIF.

    Parameters
    ----------
    figs : list[matplotlib.figure.Figure] — one figure per forecast lead time
    filename : str — output path (should end in .gif)
    duration_ms : int — milliseconds per frame

    Returns
    -------
    str — path to the saved file
    """
    # Third-party
    from PIL import Image  # Pillow is already a project dependency

    frames = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if not frames:
        return filename

    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    return filename


# ---------------------------------------------------------------------------
# Part 2.9 — Power spectrum comparison
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_power_spectrum(
    truth,
    ens_mean,
    ens_members,
    grid_shape,
    var_name: str = "",
    title=None,
):
    """Radially-averaged 2-D power spectrum comparison.

    Detects MSE-induced spectral smoothing: neural models trained with squared
    losses tend to under-represent high-wavenumber (small-scale) variability.
    Individual ensemble members should match the truth spectrum better than
    the mean, verifying that the model preserves spatial variability.

    Parameters
    ----------
    truth : (N_grid,) or (grid_x, grid_y) array-like — ground truth field
    ens_mean : same — ensemble mean field
    ens_members : (S, N_grid) or (S, grid_x, grid_y) — ensemble members
    grid_shape : (int, int) — (grid_x, grid_y) reshape dimensions
    var_name : str — variable name for the title
    """

    def _radial_power(field_2d):
        """Compute radially-averaged power spectrum of a 2-D field."""
        nx, ny = field_2d.shape
        fft2 = np.fft.rfft2(field_2d)
        power = np.abs(fft2) ** 2

        kx = np.fft.fftfreq(nx)
        ky = np.fft.rfftfreq(ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_mag = np.sqrt(KX**2 + KY**2)

        k_bins = np.linspace(0, k_mag.max(), min(nx, ny) // 2)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        power_radial, _ = np.histogram(
            k_mag.ravel(), bins=k_bins, weights=power.ravel()
        )
        counts, _ = np.histogram(k_mag.ravel(), bins=k_bins)
        power_radial = np.where(counts > 0, power_radial / counts, np.nan)
        return k_centers, power_radial

    truth_np = _to_numpy(truth).reshape(grid_shape)
    mean_np = _to_numpy(ens_mean).reshape(grid_shape)
    members_np = _to_numpy(ens_members)
    if members_np.ndim == 2:
        members_np = members_np.reshape(-1, *grid_shape)

    fig, ax = plt.subplots(figsize=(8, 5))

    k, p_truth = _radial_power(truth_np)
    ax.loglog(k, p_truth, "k-", linewidth=2, label="Truth")

    k, p_mean = _radial_power(mean_np)
    ax.loglog(k, p_mean, "b-", linewidth=2, label="Ens. Mean")

    for m_i, member in enumerate(members_np):
        k, p_member = _radial_power(member)
        ax.loglog(
            k,
            p_member,
            color="steelblue",
            linewidth=0.6,
            alpha=0.35,
            label="Members" if m_i == 0 else None,
        )

    ax.set_xlabel("Wavenumber (cycles / grid spacing)", size=_LABEL_SIZE)
    ax.set_ylabel("Power", size=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_TICK_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title(
            (
                _tex_safe(f"Power Spectrum — {var_name}")
                if var_name
                else "Power Spectrum"
            ),
            size=_TITLE_SIZE,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 2.10 — Latent-space PCA scatter
# ---------------------------------------------------------------------------


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_pca(prior_samples, vi_samples, title=None):
    """PCA scatter of prior vs variational-inference latent samples.

    Projects all samples onto the top 2 principal components computed jointly
    on prior and VI samples. Reveals whether the encoder modifies the prior
    (well-separated clouds), or adds no information (overlapping clouds).
    Uses NumPy SVD — no scikit-learn dependency.

    Parameters
    ----------
    prior_samples : (S, N_mesh, d_latent) or (S, d_latent) tensor or array
    vi_samples : same shape as prior_samples
    """
    prior_np = _to_numpy(prior_samples)
    vi_np = _to_numpy(vi_samples)

    # Aggregate spatial dim if present: (S, N_mesh, d_latent) → (S, d_latent)
    if prior_np.ndim == 3:
        prior_np = prior_np.mean(axis=1)
        vi_np = vi_np.mean(axis=1)

    # Joint PCA via truncated SVD
    all_samples = np.concatenate([prior_np, vi_np], axis=0)  # (2S, d_latent)
    mean = all_samples.mean(axis=0)
    centered = all_samples - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_coords = centered @ Vt[:2].T  # (2S, 2)

    n_prior = prior_np.shape[0]
    prior_pca = pca_coords[:n_prior]
    vi_pca = pca_coords[n_prior:]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        prior_pca[:, 0],
        prior_pca[:, 1],
        alpha=0.6,
        s=20,
        color="steelblue",
        label="Prior",
    )
    ax.scatter(
        vi_pca[:, 0],
        vi_pca[:, 1],
        alpha=0.6,
        s=20,
        color="salmon",
        label="Variational",
    )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    ax.set_xlabel("PC 1", size=_LABEL_SIZE)
    ax.set_ylabel("PC 2", size=_LABEL_SIZE)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.legend(fontsize=_TICK_SIZE)

    if title:
        ax.set_title(title, size=_TITLE_SIZE)
    else:
        ax.set_title(
            "Latent Space — PCA Scatter (Prior vs. VI)", size=_TITLE_SIZE
        )

    fig.tight_layout()
    return fig
