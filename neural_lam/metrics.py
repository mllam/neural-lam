# Third-party
import torch


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


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
    Estimator from ensemble samples.  See e.g. Weatherbench 2.

    Supports three estimator variants (see Ferro 2014,
    https://arxiv.org/html/2412.15832v1):
        - "biased"      : diff_factor = 1 / M
        - "unbiased"    : diff_factor = 1 / (M - 1)          (fair)
        - "almost-fair" : diff_factor = (M-1+α) / (M*(M-1))

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
    num_ens = pred.shape[ens_dim]  # Number of ensemble members

    # ------------------------------------------------------------------
    # S = 1 : CRPS degenerates to MAE
    # ------------------------------------------------------------------
    if num_ens == 1:
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

    assert (
        num_ens > 1
    ), "CRPS can only be estimated for ensemble with more than 1 member"

    # ------------------------------------------------------------------
    # MAE term:  E|X_i - y|
    # ------------------------------------------------------------------
    mean_mae = torch.mean(
        torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
    )  # (..., N, d_state)

    # ------------------------------------------------------------------
    # Spread term factor depends on estimator choice
    # ------------------------------------------------------------------
    if estimator == "biased":
        diff_factor = 1 / num_ens
    elif estimator == "unbiased":
        diff_factor = 1 / (num_ens - 1)
    elif estimator == "almost-fair":
        assert (
            afc_alpha is not None
        ), "afc_alpha must be provided for almost-fair CRPS estimator"
        diff_factor = (num_ens - 1 + afc_alpha) / (
            num_ens * (num_ens - 1)
        )
    else:
        raise NotImplementedError(f"Unknown CRPS estimator: {estimator}")

    # ------------------------------------------------------------------
    # S = 2 : closed-form pair difference (no sorting needed)
    # ------------------------------------------------------------------
    if num_ens == 2 and estimator == "unbiased":
        pair_diffs_term = (
            -0.5
            * diff_factor
            * torch.abs(
                pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
            )
        )  # (..., N, d_state)
    else:
        # --------------------------------------------------------------
        # Rank-based O(M log M) spread term for general case
        # Ranks start at 1; two argsorts compute entry ranks
        # --------------------------------------------------------------
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
    assert (
        pred.shape[ens_dim] > 1
    ), "spread_squared requires more than 1 ensemble member"
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

