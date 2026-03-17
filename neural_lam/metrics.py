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

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` where leading
          dimensions are broadcastable.
    mask : torch.Tensor or None
        Boolean mask selecting which grid nodes to include. Pass ``None`` to
        use all nodes.

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool
        If ``True``, reduce ``num_grid_nodes`` by taking the mean,
        producing ``(..., num_variables)``.
    sum_vars : bool
        If ``True``, reduce the variable dimension ``num_variables`` by
        summing, producing ``(..., num_grid_nodes)`` or ``(...,)`` depending on
        ``average_grid``.

    Returns
    -------
    torch.Tensor
        Reduced metric tensor.

        * **Shape**: one of ``(...,)``, ``(..., num_variables)``,
          ``(..., num_grid_nodes)``, or ``(..., num_grid_nodes, num_variables)``
          depending on the reduction flags.
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

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Predicted standard deviation used as the per-entry weighting.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
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


def mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Compute the unweighted Mean Squared Error (MSE).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Unused argument for API parity with :func:`wmse`.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        MSE after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Compute the Weighted Mean Absolute Error (wMAE).

    Scales the absolute error at each grid node and variable by the inverse
    standard deviation ``1 / pred_std``, then applies masking and reduction via
    :func:`mask_and_reduce_metric`.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Predicted standard deviation used as the per-entry weighting.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Weighted MAE after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
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


def mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Compute the unweighted Mean Absolute Error (MAE).

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Unused argument for compatibility with :func:`wmae`.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        MAE after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
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

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Predicted standard deviation parameter of the Gaussian.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Negative log-likelihood after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
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
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
):
    """
    Compute the (negative) Continuous Ranked Probability Score (CRPS).

    A closed-form expression for a Gaussian predictive distribution is used.

    Parameters
    ----------
    pred : torch.Tensor
        Distribution mean predictions.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    target : torch.Tensor
        Ground-truth values.

        * **Shape**: ``(..., num_grid_nodes, num_variables)``
    pred_std : torch.Tensor
        Predicted standard deviation parameter of the Gaussian.

        * **Shape**: ``(..., num_grid_nodes, num_variables)`` or
          ``(num_variables,)``
    mask : torch.Tensor or None, optional
        Boolean mask selecting grid nodes. Default is ``None`` (all nodes).

        * **Shape**: ``(num_grid_nodes,)``
    average_grid : bool, optional
        If ``True``, average over the grid dimension. Default is ``True``.
    sum_vars : bool, optional
        If ``True``, sum over the variable dimension. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Negative CRPS values after masking and reduction (see
        :func:`mask_and_reduce_metric`).

        * **Shape**: determined by ``average_grid`` and ``sum_vars``.
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


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
}
