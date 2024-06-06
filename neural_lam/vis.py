# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# First-party
from neural_lam import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

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
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, obs_mask, title=None, vrange=None):
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

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": constants.LAMBERT_PROJ}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
        im = ax.imshow(
            data_grid,
            origin="lower",
            extent=constants.GRID_LIMITS,
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
def plot_ensemble_prediction(
    samples, target, ens_mean, ens_std, obs_mask, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    samples: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (samples, target))
        vmax = max(vals.max().cpu().item() for vals in (samples, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 15),
        subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        target,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[1],
        ens_mean,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ens. Mean",
    )
    std_im = plot_on_axis(
        axes[2], ens_std, alpha=pixel_alpha, ax_title="Ens. Std."
    )  # Own vrange

    # Plot samples
    for member_i, (ax, member) in enumerate(
        zip(axes[3:], samples[:6]), start=1
    ):
        plot_on_axis(
            ax,
            member,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            ax_title=f"Member {member_i}",
        )

    # Turn off unused axes
    for ax in axes[(3 + samples.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:2], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


def plot_on_axis(ax, data, alpha=None, vmin=None, vmax=None, ax_title=None):
    """
    Plot weather state on given axis
    """
    ax.coastlines()  # Add coastline outlines
    data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
    im = ax.imshow(
        data_grid,
        origin="lower",
        extent=constants.GRID_LIMITS,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, title=None, vrange=None):
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

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8), subplot_kw={"projection": constants.LAMBERT_PROJ}
    )

    ax.coastlines()  # Add coastline outlines
    error_grid = error.reshape(*constants.GRID_SHAPE).cpu().numpy()

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=constants.GRID_LIMITS,
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


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """
    Plot samples of latent variable drawn from prior and
    variational distribution

    prior_samples: (samples, N_mesh, d_latent)
    vi_samples: (samples, N_mesh, d_latent)

    Returns:
    fig: the plot figure
    """
    num_samples, num_mesh_nodes, latent_dim = prior_samples.shape
    plot_dims = min(latent_dim, 3)  # Plot first 3 dimensions
    img_side_size = int(np.sqrt(num_mesh_nodes))
    assert img_side_size**2 == num_mesh_nodes, (
        "Number of mesh nodes is not a "
        "square number, can not plot latent samples as images"
    )

    # Get common scale for values
    vmin = min(
        vals[..., :plot_dims].min().cpu().item()
        for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        vals[..., :plot_dims].max().cpu().item()
        for vals in (prior_samples, vi_samples)
    )

    # Create figure
    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    # Plot samples
    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):

        for dim_i in range(plot_dims):
            prior_sample_reshaped = (
                prior_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            vi_sample_reshaped = (
                vi_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            # Plot every other as prior and vi
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_sample_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_sample_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                # Add titles at top of columns
                prior_ax.set_title(f"d{dim_i} (prior)", size=15)
                vi_ax.set_title(f"d{dim_i} (vi)", size=15)

    # Remove ticks from all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=15)

    if title:
        fig.suptitle(title, size=20)

    return fig
