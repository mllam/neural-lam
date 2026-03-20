# Third-party
import torch


def get_metric(metric_name):
    """
    Get a defined metric with given name.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    Returns
    -------
    callable
        Function implementing the metric.
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Masks and (optionally) reduces entry-wise metric values.

    Parameters
    ----------
    metric_entry_vals : torch.Tensor
        Shape (..., N, d_state), array of unaggregated metric values. (...,) 
        is any number of batch dimensions, potentially different but broadcastable.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood.

    Parameters
    ----------
    pred : torch.Tensor
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
    (Negative) Continuous Ranked Probability Score (CRPS).
    Closed-form expression based on Gaussian predictive distribution.

    Parameters
    ----------
    pred : torch.Tensor
        Shape (..., N, d_state), prediction tensor.
    target : torch.Tensor
        Shape (..., N, d_state), target tensor.
    pred_std : torch.Tensor
        Shape (..., N, d_state) or (d_state,), predicted std.-dev.
    mask : torch.Tensor, optional
        Shape (N,), boolean mask for which grid nodes to use.
    average_grid : bool, optional
        If True, reduce grid dimension -2 by mean over N.
    sum_vars : bool, optional
        If True, reduce variable dimension -1 by sum over d_state.

    Returns
    -------
    torch.Tensor
        One of (...,), (..., d_state), (..., N), (..., N, d_state),
        depending on reduction arguments.
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
