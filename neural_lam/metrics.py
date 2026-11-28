"""Evaluation metrics shared across training and validation routines."""

# Standard library
import inspect
from collections.abc import Callable
from functools import partial

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
    pred_std: torch.Tensor | None = None,
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
    pred_std : torch.Tensor or None, optional
        Unused. Accepted so that a caller holding any metric can pass one
        the same way; optional here, unlike in the metrics that score a
        distribution. Default ``None``.
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

    return mask_and_reduce_metric(
        entry_mse, mask=mask, average_grid=average_grid, sum_vars=sum_vars
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
    pred_std: torch.Tensor | None = None,
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
    pred_std : torch.Tensor or None, optional
        Unused. Accepted so that a caller holding any metric can pass one
        the same way; optional here, unlike in the metrics that score a
        distribution. Default ``None``.
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

    return mask_and_reduce_metric(
        entry_mae, mask=mask, average_grid=average_grid, sum_vars=sum_vars
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


def crps_ens(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
    ens_dim: int = 1,
    *,
    afc_alpha: float = 1.0,
    chunk_size: int | None = None,
    chunk_dim: int = 0,
) -> torch.Tensor:
    """Compute ensemble CRPS with empirical, fair or almost-fair estimation.

    Uses sorted ranks to compute the pairwise spread term with memory
    linear in the number of members, with a direct two-member formula.
    Gradients are retained for training. The almost-fair variant follows
    Lang et al. (2024), AIFS: https://arxiv.org/abs/2412.15832.

    Parameters
    ----------
    pred : torch.Tensor
        Target shape with an additional ensemble dimension of size ``S``
        at ``ens_dim``, normally ``(B, S, pred_steps, num_grid_nodes,
        num_state_vars)``. At least two ensemble members are required.
    target : torch.Tensor
        Shape ``(..., num_grid_nodes, num_state_vars)``. Ground truth.
    pred_std : torch.Tensor or None, optional
        Unused; accepted for compatibility with the shared metric interface.
    mask : torch.Tensor or None, optional
        Shape ``(num_grid_nodes,)``. Boolean mask selecting grid nodes.
    average_grid : bool, optional
        Average over selected grid nodes. Default True.
    sum_vars : bool, optional
        Sum over state variables. Default True.
    ens_dim : int, optional
        Ensemble dimension in ``pred``. Default 1.
    afc_alpha : float, optional
        Interpolates from empirical (unfair) CRPS at 0 to fair CRPS
        at 1 (default). Intermediate values give almost-fair CRPS.
    chunk_size : int or None, optional
        Maximum entries along chunk_dim per scoring call. None disables
        chunking. Smaller chunks reduce peak temporary memory; gradients
        remain available to callers using CRPS for training.
    chunk_dim : int, optional
        Dimension in pred to chunk, excluding ens_dim. Defaults to
        batch dimension 0; use 2 for forecast steps in (B, S, T, N, F).
        Negative dimension indices are supported.

    Returns
    -------
    torch.Tensor
        CRPS with the ensemble dimension removed, reduced according to
        ``mask``, ``average_grid`` and ``sum_vars`` as for ``nll``.

    Raises
    ------
    ValueError
        If alpha, ensemble layout or chunk settings are invalid.
    """
    if not 0 <= afc_alpha <= 1:
        raise ValueError("afc_alpha must be finite and between 0 and 1")
    if pred.ndim != target.ndim + 1:
        raise ValueError("CRPS predictions require an extra ensemble dimension")
    if not -pred.ndim <= ens_dim < pred.ndim:
        raise ValueError("ens_dim is outside the prediction dimensions")
    # Normalize negative axes so shape slicing and axis comparisons agree.
    ens_dim %= pred.ndim
    if pred.shape[:ens_dim] + pred.shape[ens_dim + 1 :] != target.shape:
        raise ValueError(
            "CRPS target shape must match predictions without ens_dim"
        )
    num_members = pred.shape[ens_dim]
    if num_members < 2:
        raise ValueError("CRPS requires at least two ensemble members")
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not -pred.ndim <= chunk_dim < pred.ndim:
            raise ValueError("chunk_dim is outside the prediction dimensions")
        chunk_dim %= pred.ndim
        if chunk_dim == ens_dim:
            raise ValueError("chunk_dim must differ from ens_dim")
        target_dim = chunk_dim - int(chunk_dim > ens_dim)
        entries = [
            crps_ens(
                member_chunk,
                target_chunk,
                average_grid=False,
                sum_vars=False,
                ens_dim=ens_dim,
                afc_alpha=afc_alpha,
            )
            for member_chunk, target_chunk in zip(
                pred.split(chunk_size, dim=chunk_dim),
                target.split(chunk_size, dim=target_dim),
                strict=True,
            )
        ]
        return mask_and_reduce_metric(
            torch.cat(entries, dim=target_dim), mask, average_grid, sum_vars
        )
    first = (pred - target.unsqueeze(ens_dim)).abs().mean(dim=ens_dim)
    if num_members == 2:
        spread = (pred.select(ens_dim, 0) - pred.select(ens_dim, 1)).abs() / 2
    else:
        ordered = pred.sort(dim=ens_dim).values
        rank_shape = [1] * pred.ndim
        rank_shape[ens_dim] = num_members
        ranks = torch.arange(
            1,
            num_members + 1,
            device=pred.device,
            dtype=pred.dtype,
        ).view(rank_shape)
        spread = ((2 * ranks - num_members - 1) * ordered).sum(dim=ens_dim)
        spread = spread / (num_members * (num_members - 1))
    spread = spread * ((num_members - 1 + afc_alpha) / num_members)
    entry_crps = first - spread
    return mask_and_reduce_metric(
        entry_crps, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


crps_ens_fair = partial(crps_ens, afc_alpha=1.0)
crps_ens_unfair = partial(crps_ens, afc_alpha=0.0)
crps_ens_almost_fair = partial(crps_ens, afc_alpha=0.95)

ENSEMBLE_METRICS: dict[str, Callable[..., torch.Tensor]] = {
    "crps_ens": crps_ens,
    "crps_ens_fair": crps_ens_fair,
    "crps_ens_unfair": crps_ens_unfair,
    "crps_ens_almost_fair": crps_ens_almost_fair,
}


DEFINED_METRICS: dict[str, Callable[..., torch.Tensor]] = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    **ENSEMBLE_METRICS,
}


def requires_pred_std(metric: Callable[..., torch.Tensor]) -> bool:
    """
    Return whether ``metric`` needs a ``pred_std`` to be computed.

    Lets a caller holding a metric decide whether it has to come up with a
    standard deviation at all, rather than assuming every metric uses one.
    Read off the signature, so declaring ``pred_std`` without a default is
    the only place a metric states that it scores a distribution.

    Parameters
    ----------
    metric : callable
        A metric from ``DEFINED_METRICS``, e.g. as returned by
        ``get_metric``.

    Returns
    -------
    bool
        True if ``metric`` takes ``pred_std`` as a required argument.
    """
    pred_std_param = inspect.signature(metric).parameters["pred_std"]
    return pred_std_param.default is inspect.Parameter.empty


def requires_ensemble(metric: Callable[..., torch.Tensor]) -> bool:
    """Return whether a registered metric scores an ensemble of forecasts.

    Parameters
    ----------
    metric : callable
        A function returned by get_metric.

    Returns
    -------
    bool
        True for registered ensemble metrics, which require predictions
        with shape (B, S, pred_steps, num_grid_nodes, num_state_vars).
    """
    return metric in ENSEMBLE_METRICS.values()
