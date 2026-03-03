# Third-party
import torch

def get_metric(metric_name):
    """
    Retrieves a metric function by its name.

    Args:
        metric_name (str): Name of the metric (e.g., 'mse', 'wmse', 'nll').

    Returns:
        function: The corresponding function implementing the requested metric.

    Raises:
        AssertionError: If the metric_name is not found in DEFINED_METRICS.
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Applies spatial masking and reduces dimensions of the metric values.

    Args:
        metric_entry_vals (torch.Tensor): Entry-wise values, shape (..., N, d_state).
        mask (torch.Tensor, optional): Boolean mask for grid nodes, shape (N,).
        average_grid (bool): If True, reduces the grid dimension (mean over N).
        sum_vars (bool): If True, reduces the variable dimension (sum over d_state).

    Returns:
        torch.Tensor: Reduced metric value. Shape depends on reduction flags:
            - (...,) if both average_grid and sum_vars are True.
            - (..., d_state) if only average_grid is True.
            - (..., N) if only sum_vars is True.
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
    Computes the Weighted Mean Squared Error.
    Useful when certain nodes or variables have different error scales.

    Args:
        pred (torch.Tensor): Prediction tensor, shape (..., N, d_state).
        target (torch.Tensor): Ground truth tensor, shape (..., N, d_state).
        pred_std (torch.Tensor): Predicted standard deviation for weighting, 
            shape (..., N, d_state) or (d_state,).
        mask (torch.Tensor, optional): Spatial node mask, shape (N,).
        average_grid (bool): Whether to average across the grid nodes.
        sum_vars (bool): Whether to sum across the state variables.

    Returns:
        torch.Tensor: The computed WMSE value.
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
    Computes the standard Unweighted Mean Squared Error.
    Internal call to wmse with unit weights.

    Args:
        pred, target, pred_std, mask, average_grid, sum_vars: See wmse.

    Returns:
        torch.Tensor: The computed MSE value.
    """
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Computes the Weighted Mean Absolute Error.

    Args:
        pred, target, pred_std, mask, average_grid, sum_vars: See wmse.

    Returns:
        torch.Tensor: The computed WMAE value.
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
    Computes the standard Unweighted Mean Absolute Error.

    Args:
        pred, target, pred_std, mask, average_grid, sum_vars: See wmse.

    Returns:
        torch.Tensor: The computed MAE value.
    """
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Computes the Negative Log Likelihood loss for an isotropic Gaussian likelihood.
    Useful for probabilistic forecasting models.

    Args:
        pred (torch.Tensor): Mean of the Gaussian, shape (..., N, d_state).
        target (torch.Tensor): Target values, shape (..., N, d_state).
        pred_std (torch.Tensor): Std-dev of the Gaussian, shape (..., N, d_state).

    Returns:
        torch.Tensor: Scalar NLL value after masking and reduction.
    """
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def crps_gauss(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
):
    """
    Computes the Negative Continuous Ranked Probability Score (CRPS).
    Uses the closed-form expression for a Gaussian predictive distribution.

    Args:
        pred (torch.Tensor): Predictive mean, shape (..., N, d_state).
        target (torch.Tensor): Observation, shape (..., N, d_state).
        pred_std (torch.Tensor): Predictive standard deviation.

    Returns:
        torch.Tensor: The computed CRPS value.
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