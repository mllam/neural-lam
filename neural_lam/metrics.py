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
    assert metric_name_lower in DEFINED_METRICS, (
        f"Unknown metric: {metric_name}"
    )
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
    if mask is not None:
        rows = torch.where(mask.any(dim=-1))[0]
        cols = torch.where(mask.any(dim=-2))[0]
        if len(rows) > 0 and len(cols) > 0:
            field_input = field_2d[
                ..., rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1
            ]
        else:
            field_input = field_2d
    else:
        field_input = field_2d

    h = field_input.shape[-2]
    w = field_input.shape[-1]
    dct_engine = DiscreteCosineTransform2D(h, w, device=field_2d.device)

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
    if kernel_size % 2 == 0:
        raise ValueError(
            f"Fractions skill score kernel_size must be odd, got {kernel_size}."
        )

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
        mask_in = mask.unsqueeze(0).unsqueeze(0).float()
        # Require entire neighborhood kernel to lie inside interior mask
        valid_mask = (
            F.conv2d(mask_in, kernel, padding=padding) >= 1.0 - 1e-6
        ).float()
        diff_sq = diff_sq * valid_mask
        ref_sq = ref_sq * valid_mask
        mask_sum = valid_mask.sum() * diff_sq.shape[0]
        mse = diff_sq.sum() / (mask_sum + 1e-8)
        ref = ref_sq.sum() / (mask_sum + 1e-8)
    else:
        mse = diff_sq.mean()
        ref = ref_sq.mean()

    return 1.0 - (mse / (ref + 1e-8))


def log_spectral_distance(
    psd_model: torch.Tensor,
    psd_target: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Log-Spectral Distance (LSD) between model and target power spectra.

    LSD (dB) = sqrt( mean( (10 * log10( psd_target / psd_model ))^2 ) )

    Parameters
    ----------
    psd_model : torch.Tensor
        Model power spectral density. Shape ``(..., K)``.
    psd_target : torch.Tensor
        Target / ground-truth power spectral density. Shape ``(..., K)``.
    eps : float, default 1e-10
        Small constant to avoid division by zero or log of non-positive numbers.

    Returns
    -------
    torch.Tensor
        Log-spectral distance in decibels (dB). Shape ``(...)``.
    """
    ratio = torch.clamp(psd_target, min=eps) / torch.clamp(psd_model, min=eps)
    log_ratio = 10.0 * torch.log10(ratio)
    return torch.sqrt(torch.mean(log_ratio**2, dim=-1))


def contingency_table(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hits, false alarms, misses, and correct negatives for a threshold.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.
    target : torch.Tensor
        Ground-truth targets.
    threshold : float
        Event exceedance threshold.
    mask : torch.Tensor or None, optional
        Boolean spatial mask.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        Hits (H), False Alarms (F), Misses (M), Correct Negatives (CN).
    """
    pred_event = pred >= threshold
    targ_event = target >= threshold

    if mask is not None:
        pred_event = pred_event[mask]
        targ_event = targ_event[mask]

    hits = (pred_event & targ_event).sum().float()
    false_alarms = (pred_event & ~targ_event).sum().float()
    misses = (~pred_event & targ_event).sum().float()
    correct_negatives = (~pred_event & ~targ_event).sum().float()

    return hits, false_alarms, misses, correct_negatives


def frequency_bias(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Frequency Bias (FBIAS = (Hits + False Alarms) / (Hits + Misses)).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.
    target : torch.Tensor
        Ground-truth targets.
    threshold : float
        Event exceedance threshold.
    mask : torch.Tensor or None, optional
        Boolean spatial mask.

    Returns
    -------
    torch.Tensor
        Frequency Bias ratio. 1.0 indicates unbiased frequency.
    """
    h, f, m, _ = contingency_table(pred, target, threshold, mask=mask)
    return (h + f) / (h + m + 1e-8)


def equitable_threat_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Equitable Threat Score (Gilbert Skill Score).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.
    target : torch.Tensor
        Ground-truth targets.
    threshold : float
        Event exceedance threshold.
    mask : torch.Tensor or None, optional
        Boolean spatial mask.

    Returns
    -------
    torch.Tensor
        Equitable Threat Score in [-1/3, 1].
    """
    h, f, m, cn = contingency_table(pred, target, threshold, mask=mask)
    total = h + f + m + cn
    hits_random = ((h + m) * (h + f)) / (total + 1e-8)
    return (h - hits_random) / (h + m + f - hits_random + 1e-8)


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
}
