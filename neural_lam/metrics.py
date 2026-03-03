# Third-party
import torch


def get_metric(metric_name):
    """Get a defined metric by name.

    Args:
        metric_name (str): Name of the metric.

    Returns:
        Callable: Function implementing the selected metric.

    Raises:
        AssertionError: If the metric name is not defined.
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """Mask and optionally reduce entry-wise metric values.

    (...,) represents any number of batch dimensions, potentially
    different but broadcastable.

    Args:
        metric_entry_vals (torch.Tensor): Tensor of shape (..., N, d_state).
        mask (torch.Tensor or None): Boolean mask of shape (N,) describing
            which grid nodes to include in the metric.
        average_grid (bool): If True, reduce grid dimension (-2)
            by taking the mean over N.
        sum_vars (bool): If True, reduce variable dimension (-1)
            by summing over d_state.

    Returns:
        torch.Tensor: One of (...,), (..., d_state), (..., N), or
        (..., N, d_state), depending on reduction arguments.
    """
    if mask is not None:
        metric_entry_vals = metric_entry_vals[..., mask, :]

    if average_grid:
        metric_entry_vals = torch.mean(metric_entry_vals, dim=-2)

    if sum_vars:
        metric_entry_vals = torch.sum(metric_entry_vals, dim=-1)

    return metric_entry_vals


def wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """Weighted Mean Squared Error.

    (...,) represents any number of batch dimensions, potentially
    different but broadcastable.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Predicted standard deviation of shape
            (..., N, d_state) or (d_state,).
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )
    entry_mse_weighted = entry_mse / (pred_std**2)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """Unweighted Mean Squared Error.

    Equivalent to weighted MSE with unit standard deviation.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Ignored; replaced with ones.
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """Weighted Mean Absolute Error.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Predicted standard deviation.
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )
    entry_mae_weighted = entry_mae / pred_std

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """Unweighted Mean Absolute Error.

    Equivalent to weighted MAE with unit standard deviation.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Ignored; replaced with ones.
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """Negative Log Likelihood for isotropic Gaussian likelihood.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Predicted standard deviation.
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    dist = torch.distributions.Normal(pred, pred_std)
    entry_nll = -dist.log_prob(target)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def crps_gauss(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
):
    """Negative Continuous Ranked Probability Score (CRPS).

    Closed-form expression based on Gaussian predictive distribution.

    Args:
        pred (torch.Tensor): Predictions of shape (..., N, d_state).
        target (torch.Tensor): Targets of shape (..., N, d_state).
        pred_std (torch.Tensor): Predicted standard deviation.
        mask (torch.Tensor, optional): Boolean mask of shape (N,).
        average_grid (bool): If True, average over grid dimension.
        sum_vars (bool): If True, sum over variable dimension.

    Returns:
        torch.Tensor: Metric value depending on reduction arguments.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device),
        torch.ones((), device=pred.device),
    )
    target_standard = (target - pred) / pred_std

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )

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
