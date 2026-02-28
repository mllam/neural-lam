# Third-party
import torch




def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
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
    Weighted Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
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
    (Unweighted) Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
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
    (Unweighted) Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
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
    (Negative) Continuous Ranked Probability Score (CRPS)
    Closed-form expression based on Gaussian predictive distribution

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
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


def crps_ensemble(predictions, target):
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    Computes the energy-form CRPS, which measures the quality of probabilistic
    predictions using an ensemble of members.

    Parameters
    ----------
    predictions : torch.Tensor
        Ensemble predictions of shape (E, B, T, N, F), where E is ensemble
        size, B is batch, T is time steps, N is grid nodes, F is features.
    target : torch.Tensor
        Ground truth tensor of shape (B, T, N, F).

    Returns
    -------
    torch.Tensor
        CRPS tensor of shape (B, T, N, F). Lower is better.
    """
    # term1: mean absolute error between each ensemble member and target
    term1 = torch.mean(torch.abs(predictions - target.unsqueeze(0)), dim=0)

    # term2: mean pairwise spread across ensemble members
    pairwise_diff = torch.abs(
        predictions.unsqueeze(0) - predictions.unsqueeze(1)
    )
    term2 = torch.mean(pairwise_diff, dim=(0, 1)) / 2

    return term1 - term2


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "crps": crps_ensemble,
}
