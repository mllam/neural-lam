"""Evaluation metrics shared across training and validation routines."""

# Third-party
import torch


def get_metric(metric_name):
    """
    Retrieve a registered metric function by name.

    Parameters
    ----------
    metric_name : str
        Name of the metric to load (case-insensitive).

    Returns
    -------
    callable
        Metric function implementing the requested metric.

    Raises
    ------
    AssertionError
        If ``metric_name`` is not part of :data:`DEFINED_METRICS`.
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Apply a spatial mask and optionally reduce a per-entry metric tensor.

    Parameters
    ----------
    metric_entry_vals : torch.Tensor
        Entry-wise metric values.

        * **Shape**: ``(..., N, d_state)`` where ``...`` are broadcastable
          leading dimensions.
    mask : torch.Tensor or None
        Boolean mask selecting which grid nodes to include. Pass ``None`` to
        use all nodes.

        * **Shape**: ``(N,)``
    average_grid : bool
        If ``True``, reduce the grid dimension ``N`` by taking the mean,
        producing ``(..., d_state)``.
    sum_vars : bool
        If ``True``, reduce the variable dimension ``d_state`` by summing,
        producing ``(..., N)`` or ``(...,)`` depending on ``average_grid``.

    Returns
    -------
    torch.Tensor
        Reduced metric tensor.

        * **Shape**: one of ``(...,)``, ``(..., d_state)``, ``(..., N)``, or
          ``(..., N, d_state)`` depending on the reduction flags.
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
    Compute the Weighted Mean Squared Error (wMSE).

    Scales the squared error at each grid node and variable by the inverse
    variance ``1 / pred_std**2``, then applies masking and reduction via
    :func:`mask_and_reduce_metric`.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Predicted standard deviation used as the per-entry weighting.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Weighted MSE after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
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
    Compute the unweighted Mean Squared Error (MSE).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Unused argument for API parity with :func:`wmse`.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        MSE with shape determined by ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Compute the Weighted Mean Absolute Error (wMAE).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Predicted standard deviation used as the per-entry weighting.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Weighted MAE with shape determined by ``average_grid`` and
        ``sum_vars``.
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
    Compute the unweighted Mean Absolute Error (MAE).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Unused argument for compatibility with :func:`wmae`.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        MAE with shape determined by ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Compute the Negative Log Likelihood for an isotropic Gaussian likelihood.

    Parameters
    ----------
    pred : torch.Tensor
        Distribution mean predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Predicted standard deviation parameter of the Gaussian.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Negative log-likelihood with shape determined by ``average_grid`` and
        ``sum_vars``.
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
    Compute the (negative) Continuous Ranked Probability Score (CRPS).

    A closed-form expression for a Gaussian predictive distribution is used.

    Parameters
    ----------
    pred : torch.Tensor
        Distribution mean predictions.

        * **Shape**: ``(..., N, d_state)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., N, d_state)``
    pred_std : torch.Tensor
        Predicted standard deviation parameter of the Gaussian.

        * **Shape**: ``(..., N, d_state)`` or ``(d_state,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(N,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Negative CRPS values with shape determined by ``average_grid`` and
        ``sum_vars``.
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
