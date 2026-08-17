"""Evaluation metrics shared across training and validation routines."""

# Standard library
import math
from collections.abc import Callable

# Third-party
import torch
import torch.nn.functional as F


def get_metric(metric_name: str) -> Callable[..., torch.Tensor]:
    """
    Get a metric function by name.

    Parameters
    ----------
    metric_name : str
        Name of the metric. Must be a key in ``DEFINED_METRICS``.

    Returns
    -------
    callable
        Function implementing the requested metric.

    Raises
    ------
    AssertionError
        If ``metric_name`` (case-insensitive) is not a key in
        ``DEFINED_METRICS``.
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(
    metric_entry_vals: torch.Tensor,
    mask: torch.Tensor | None,
    average_grid: bool,
    sum_vars: bool,
) -> torch.Tensor:
    """
    Apply a boolean mask and optionally reduce a per-entry metric tensor.

    Parameters
    ----------
    metric_entry_vals : torch.Tensor
        Shape ``(..., N, num_variables)``. Per-entry metric values. ``(...)``
        denotes any number of broadcastable batch dimensions, ``N`` is
        the number of grid nodes, and ``num_variables`` is the number of
        variables in the gridded representation (e.g. state features).
    mask : torch.Tensor or None
        Shape ``(N,)``. Boolean mask selecting which grid nodes to
        include. ``None`` means all nodes are used.
    average_grid : bool
        If True, average over the grid dimension ``N``.
    sum_vars : bool
        If True, sum over the variable dimension ``num_variables``.

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[
            ..., mask, :
        ]  # (..., num_selected_nodes, num_variables)

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.mean(
            metric_entry_vals, dim=-2
        )  # (..., num_variables)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., num_grid_nodes) or (...,)

    return metric_entry_vals


def wmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Weighted Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation used as per-entry weight.
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., num_grid_nodes, num_variables)
    entry_mse_weighted = entry_mse / (
        pred_std**2
    )  # (..., num_grid_nodes, num_variables)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    (Unweighted) Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation (unused; ``pred_std`` is replaced by ones
        internally).
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Weighted Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation used as per-entry weight.
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., num_grid_nodes, num_variables)
    entry_mae_weighted = (
        entry_mae / pred_std
    )  # (..., num_grid_nodes, num_variables)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    (Unweighted) Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation (unused; ``pred_std`` is replaced by ones
        internally).
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Negative Log Likelihood loss for an isotropic Gaussian likelihood.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Predicted mean. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation of the Gaussian.
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Broadcast pred_std if shaped (num_variables,) via distribution internals
    dist = torch.distributions.Normal(
        pred, pred_std
    )  # (..., num_grid_nodes, num_variables)
    entry_nll = -dist.log_prob(target)  # (..., num_grid_nodes, num_variables)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def crps_gauss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (CRPS) for a Gaussian predictive
    distribution (closed-form expression, negated for minimisation).

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Predicted mean. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation of the Gaussian.
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (
        target - pred
    ) / pred_std  # (..., num_grid_nodes, num_variables)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., num_grid_nodes, num_variables)

    return mask_and_reduce_metric(
        entry_crps, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def mbe(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Mean Bias Error (predicted - target).

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, num_variables)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``num_variables`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, num_variables)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, num_variables)`` or ``(num_variables,)``. Predicted
        standard deviation (unused, kept for signature consistency).
    mask : torch.Tensor or None, optional
        Shape ``(N,)``. Boolean mask over grid nodes. ``None`` uses all
        nodes.
    average_grid : bool, optional
        If True, average over the grid dimension (default True).
    sum_vars : bool, optional
        If True, sum over the variable dimension (default True).

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., num_variables)``, ``(..., N)``, or ``(..., N, num_variables)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    entry_bias = pred - target  # (..., num_grid_nodes, num_variables)
    return mask_and_reduce_metric(
        entry_bias, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def radial_power_spectrum_2d(
    field_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the 1D radially averaged power spectral density (PSD) from 2D
    spatial fields using 2D FFT.

    Parameters
    ----------
    field_2d : torch.Tensor
        Spatial field with shape ``(..., Ny, Nx)``.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        - ``radial_psd``: Radially averaged power spectrum with shape
          ``(..., num_wavenumbers)``.
        - ``wavenumbers``: 1D tensor of integer wavenumber bin centers.
    """
    ny = field_2d.shape[-2]
    nx = field_2d.shape[-1]

    # Compute 2D discrete Fourier transform and shift zero frequency to center
    fft_2d = torch.fft.fft2(field_2d)
    fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))
    psd_2d = (torch.abs(fft_shifted) ** 2) / (ny * nx)

    # Coordinate grids centered at zero frequency
    y_freq = torch.arange(-ny // 2, ny - ny // 2, device=field_2d.device)
    x_freq = torch.arange(-nx // 2, nx - nx // 2, device=field_2d.device)
    grid_y, grid_x = torch.meshgrid(y_freq, x_freq, indexing="ij")
    radial_dist = torch.sqrt(grid_y.float() ** 2 + grid_x.float() ** 2)

    max_k = int(min(ny, nx) // 2)
    k_bins = torch.arange(1, max_k + 1, device=field_2d.device)
    radial_dist_int = torch.round(radial_dist).long()

    # Average PSD over concentric wavenumber rings
    psd_flat = psd_2d.reshape(-1, ny * nx)
    dist_flat = radial_dist_int.flatten()

    radial_psd_list = []
    for k in k_bins:
        ring_mask = dist_flat == k
        if ring_mask.any():
            ring_mean = psd_flat[:, ring_mask].mean(dim=-1)
        else:
            ring_mean = torch.zeros(
                psd_flat.shape[0], device=field_2d.device, dtype=field_2d.dtype
            )
        radial_psd_list.append(ring_mean)

    radial_psd_2d = torch.stack(radial_psd_list, dim=-1)
    batch_shape = field_2d.shape[:-2]
    radial_psd = radial_psd_2d.reshape(*batch_shape, max_k)

    return radial_psd, k_bins


class DiscreteCosineTransform2D:
    """
    Computes 2D Discrete Cosine Transform (DCT-II) for regional fields.

    Avoids FFT edge-step discontinuities on limited area domains.
    """

    def __init__(
        self, height: int, width: int, device: torch.device | str = "cpu"
    ) -> None:
        """
        Initialize the 2D DCT-II transform engine.

        Parameters
        ----------
        height : int
            Height of the spatial domain.
        width : int
            Width of the spatial domain.
        device : torch.device or str, default "cpu"
            Computation device.
        """
        self.height = height
        self.width = width
        self.device = torch.device(device)

        # Precompute DCT-II orthogonal basis matrices
        i = torch.arange(self.height, device=self.device).unsqueeze(1)
        u = torch.arange(self.height, device=self.device).unsqueeze(0)
        dct_m = torch.cos(math.pi * (2 * i + 1) * u / (2 * self.height))
        dct_m[:, 0] *= 1.0 / math.sqrt(2.0)
        dct_m *= math.sqrt(2.0 / self.height)
        self.dct_m = dct_m

        j = torch.arange(self.width, device=self.device).unsqueeze(1)
        v = torch.arange(self.width, device=self.device).unsqueeze(0)
        dct_n = torch.cos(math.pi * (2 * j + 1) * v / (2 * self.width))
        dct_n[:, 0] *= 1.0 / math.sqrt(2.0)
        dct_n *= math.sqrt(2.0 / self.width)
        self.dct_n = dct_n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT-II transform to tensor with shape ``(..., H, W)``.
        """
        x_dev = x.to(self.device)
        out = torch.matmul(self.dct_m.T, x_dev)
        out = torch.matmul(out, self.dct_n)
        return out


def dct_power_spectrum_2d(
    field_2d: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D DCT-II radially averaged power spectral density across isotropic
    wavenumber bins.

    Parameters
    ----------
    field_2d : torch.Tensor
        Spatial field with shape ``(..., H, W)``.
    mask : torch.Tensor or None, optional
        Binary interior domain mask of shape ``(H, W)``.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        - ``k_centers``: 1D tensor of normalized wavenumber centers in [0, 0.5].
        - ``radial_psd``: 1D or batched tensor of power spectral density.
    """
    h = field_2d.shape[-2]
    w = field_2d.shape[-1]
    dct_engine = DiscreteCosineTransform2D(h, w, device=field_2d.device)

    if mask is not None:
        field_input = field_2d * mask
    else:
        field_input = field_2d

    dct_coeffs = dct_engine(field_input)
    power = dct_coeffs**2

    u = torch.arange(h, device=field_2d.device).unsqueeze(1).repeat(1, w)
    v = torch.arange(w, device=field_2d.device).unsqueeze(0).repeat(h, 1)
    k_map = torch.sqrt((u / (2.0 * h)) ** 2 + (v / (2.0 * w)) ** 2)

    num_bins = min(h, w) // 2
    k_bins = torch.linspace(0.0, 0.5, num_bins + 1, device=field_2d.device)

    # Flatten spatial dims for binning
    power_flat = power.reshape(-1, h * w)
    k_flat = k_map.flatten()

    psd_list = []
    for idx in range(num_bins):
        bin_mask = (k_flat >= k_bins[idx]) & (k_flat < k_bins[idx + 1])
        if bin_mask.any():
            bin_mean = power_flat[:, bin_mask].mean(dim=-1)
        else:
            bin_mean = torch.zeros(
                power_flat.shape[0],
                device=field_2d.device,
                dtype=field_2d.dtype,
            )
        psd_list.append(bin_mean)

    radial_psd_stack = torch.stack(psd_list, dim=-1)
    batch_shape = field_2d.shape[:-2]
    radial_psd = radial_psd_stack.reshape(*batch_shape, num_bins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    return k_centers, radial_psd


def fractions_skill_score_2d(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    threshold: float,
    kernel_size: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scale-dependent Fractions Skill Score (FSS) at a specified
    exceedance threshold and spatial neighborhood kernel size.

    Parameters
    ----------
    pred_field : torch.Tensor
        Predicted field with shape ``(H, W)`` or ``(B, H, W)``.
    target_field : torch.Tensor
        Target field with shape ``(H, W)`` or ``(B, H, W)``.
    threshold : float
        Physical exceedance threshold (e.g. 10.0 m/s for wind speed).
    kernel_size : int
        Spatial neighborhood kernel size (e.g. 1, 3, 7, 15, 31). Must be odd.
    mask : torch.Tensor or None, optional
        Interior domain mask of shape ``(H, W)``.

    Returns
    -------
    torch.Tensor
        Fractions Skill Score in [0, 1].
    """
    if pred_field.ndim == 2:
        pred_in = pred_field.unsqueeze(0).unsqueeze(0)
        target_in = target_field.unsqueeze(0).unsqueeze(0)
    elif pred_field.ndim == 3:
        pred_in = pred_field.unsqueeze(1)
        target_in = target_field.unsqueeze(1)
    else:
        pred_in = pred_field
        target_in = target_field

    binary_pred = (pred_in >= threshold).float()
    binary_target = (target_in >= threshold).float()

    padding = kernel_size // 2
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size), device=pred_field.device
    ) / (kernel_size**2)

    frac_pred = F.conv2d(binary_pred, kernel, padding=padding)
    frac_target = F.conv2d(binary_target, kernel, padding=padding)

    diff_sq = (frac_pred - frac_target) ** 2
    ref_sq = (frac_pred**2) + (frac_target**2)

    if mask is not None:
        mask_in = mask.unsqueeze(0).unsqueeze(0)
        diff_sq = diff_sq * mask_in
        ref_sq = ref_sq * mask_in
        mask_sum = mask_in.sum()
        mse = diff_sq.sum() / (mask_sum + 1e-8)
        ref = ref_sq.sum() / (mask_sum + 1e-8)
    else:
        mse = diff_sq.mean()
        ref = ref_sq.mean()

    return 1.0 - (mse / (ref + 1e-8))


def spectral_collapse_ratio(
    psd_model: torch.Tensor,
    psd_target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Spectral Collapse Ratio (SCR = PSD_model / PSD_target).
    Values < 0.5 indicate severe spatial blurring/smoothing.

    Parameters
    ----------
    psd_model : torch.Tensor
        Model power spectral density.
    psd_target : torch.Tensor
        Target / ground-truth power spectral density.

    Returns
    -------
    torch.Tensor
        Ratio of model to target spectral power.
    """
    return psd_model / (psd_target + 1e-8)


def hallucination_index(
    scr_fine: float | torch.Tensor,
    fss_fine: float | torch.Tensor,
) -> float | torch.Tensor:
    """
    Compute Hallucination Index (HI = SCR_fine * (1.0 - FSS_fine)).
    High spectral energy combined with low localized spatial skill flags
    mislocated / generative hallucinations.

    Parameters
    ----------
    scr_fine : float or torch.Tensor
        Spectral Collapse Ratio at fine scales.
    fss_fine : float or torch.Tensor
        Fractions Skill Score at fine scales.

    Returns
    -------
    float or torch.Tensor
        Hallucination Index in [0, 1+].
    """
    return scr_fine * (1.0 - fss_fine)


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "mbe": mbe,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
}
