# Third-party
import torch


def get_metric(metric_name):
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
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Apply a boolean mask and optionally reduce a per-entry metric tensor.

    Parameters
    ----------
    metric_entry_vals : torch.Tensor
        Shape ``(..., N, d_state)``. Per-entry metric values. ``(...)``
        denotes any number of broadcastable batch dimensions, ``N`` is
        the number of grid nodes, and ``d_state`` is the number of state
        variables.
    mask : torch.Tensor or None
        Shape ``(N,)``. Boolean mask selecting which grid nodes to
        include. ``None`` means all nodes are used.
    average_grid : bool
        If True, average over the grid dimension ``N``.
    sum_vars : bool
        If True, sum over the variable dimension ``d_state``.

    Returns
    -------
    torch.Tensor
        Reduced metric values. Shape is one of ``(...,)``,
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[
            ..., mask, :
        ]  # (..., N', d_state)

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.mean(
            metric_entry_vals, dim=-2
        )  # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., N) or (...,)

    return metric_entry_vals


def wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mae_weighted = entry_mae / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Model prediction. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Negative Log Likelihood loss for an isotropic Gaussian likelihood.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Predicted mean. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def crps_gauss(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
):
    """
    Continuous Ranked Probability Score (CRPS) for a Gaussian predictive
    distribution (closed-form expression, negated for minimisation).

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(..., N, d_state)``. Predicted mean. ``(...)`` denotes
        any number of broadcastable batch dimensions, ``N`` is the number
        of grid nodes, and ``d_state`` is the number of state variables.
    target : torch.Tensor
        Shape ``(..., N, d_state)``. Ground-truth target. Dims: same as
        ``pred``.
    pred_std : torch.Tensor
        Shape ``(..., N, d_state)`` or ``(d_state,)``. Predicted
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
        ``(..., d_state)``, ``(..., N)``, or ``(..., N, d_state)``
        depending on ``average_grid`` and ``sum_vars``.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (target - pred) / pred_std  # (..., N, d_state)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_crps, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
}
