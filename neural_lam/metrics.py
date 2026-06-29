"""Evaluation metrics shared across training and validation routines."""

# Standard library
from collections.abc import Callable
from typing import Optional

# Third-party
import torch


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
    mask: Optional[torch.Tensor],
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
    mask: Optional[torch.Tensor] = None,
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
    mask: Optional[torch.Tensor] = None,
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
    mask: Optional[torch.Tensor] = None,
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
    mask: Optional[torch.Tensor] = None,
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
    mask: Optional[torch.Tensor] = None,
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
    mask: Optional[torch.Tensor] = None,
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


def crps_ens(
    pred,
    target,
    pred_std=None,
    mask=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
    estimator="unbiased",
    afc_alpha=None,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Estimator from ensemble samples. See e.g. Weatherbench 2.

    Supports three estimator variants:
        - "biased"      : diff_factor = 1 / M
        - "unbiased"    : diff_factor = 1 / (M - 1)          (fair)
        - "almost-fair" : diff_factor = (M-1+α) / (M*(M-1))
          (see Lang et al., 2024 — AIFS)

    Uses a rank-based O(M log M) implementation for the spread term
    instead of the O(M²) pairwise formulation (see Zamo & Naveau, WB2).

    pred:        (..., M, ..., N, d_state), ensemble predictions
    target:      (..., N, d_state), observation / ground truth
    pred_std:    unused (kept for API compatibility with other metrics)
    mask:        (N,), boolean mask for grid nodes
    average_grid: reduce grid dim -2 (mean over N)
    sum_vars:    reduce var dim -1 (sum over d_state)
    ens_dim:     dimension along which ensemble members are laid out
    estimator:   one of "biased", "unbiased", "almost-fair"
    afc_alpha:   alpha parameter, required when estimator="almost-fair"

    Returns:
    metric_val: shape depends on reduction arguments.
    """
    import warnings

    num_ens = pred.shape[ens_dim]

    # S = 1: CRPS degenerates to MAE
    if num_ens == 1:
        warnings.warn(
            "crps_ens called with a single ensemble member (S=1). "
            "Falling back to MAE. This may indicate an error in the "
            "ensemble dimension.",
            stacklevel=2,
        )
        return mae(
            pred.squeeze(ens_dim),
            target,
            pred_std.squeeze(ens_dim)
            if pred_std is not None
            else torch.ones_like(target),
            mask=mask,
            average_grid=average_grid,
            sum_vars=sum_vars,
        )

    # MAE term: E|X_i - y|
    mean_mae = torch.mean(
        torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
    )  # (..., N, d_state)

    # Spread term factor depends on estimator choice
    if estimator == "biased":
        diff_factor = 1 / num_ens
    elif estimator == "unbiased":
        diff_factor = 1 / (num_ens - 1)
    elif estimator == "almost-fair":
        if afc_alpha is None:
            raise ValueError(
                "afc_alpha must be provided for almost-fair CRPS estimator"
            )
        diff_factor = (num_ens - 1 + afc_alpha) / (
            num_ens * (num_ens - 1)
        )
    else:
        raise NotImplementedError(f"Unknown CRPS estimator: {estimator}")

    # S = 2: closed-form pair difference (no sorting needed)
    if num_ens == 2:
        pair_diffs_term = (
            -0.5
            * diff_factor
            * torch.abs(
                pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
            )
        )  # (..., N, d_state)
    else:
        # Rank-based O(M log M) spread term
        # Ranks start at 1; two argsorts compute entry ranks
        ranks = (
            pred.argsort(dim=ens_dim).argsort(dim=ens_dim) + 1
        )

        pair_diffs_term = diff_factor * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )  # (..., N, d_state)

    crps_estimator = mean_mae + pair_diffs_term  # (..., N, d_state)

    return mask_and_reduce_metric(
        crps_estimator, mask, average_grid, sum_vars
    )


def spread_squared(
    pred,
    target,  # pylint: disable=unused-argument
    pred_std=None,  # pylint: disable=unused-argument
    mask=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
):
    """
    (Squared) spread of ensemble.
    Similarly to RMSE, we want to take sqrt after spatial and sample averaging,
    so we need to average the squared spread.

    pred:        (..., M, ..., N, d_state), ensemble predictions
    target:      (..., N, d_state), observation (unused)
    pred_std:    unused (kept for API compatibility)
    mask:        (N,), boolean mask for grid nodes
    average_grid: reduce grid dim -2 (mean over N)
    sum_vars:    reduce var dim -1 (sum over d_state)
    ens_dim:     dimension along which ensemble members are laid out

    Returns:
    metric_val: depends on reduction arguments.
    """
    if pred.shape[ens_dim] <= 1:
        raise ValueError(
            "spread_squared requires more than 1 ensemble member"
        )
    entry_var = torch.var(pred, dim=ens_dim)  # (..., N, d_state)
    return mask_and_reduce_metric(entry_var, mask, average_grid, sum_vars)


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "crps_ens": crps_ens,
    "spread_squared": spread_squared,
}

