"""
verification metrics for global/regional weather model evaluation.

implements standard NWP verification scores (ACC, weighted RMSE, etc.)
that complement the training-time metrics in neural_lam.metrics.

tensor conventions match neural_lam.metrics:
    pred/target: (..., N, d_state)
    grid_weights: (N,)
    mask: (N,) boolean
"""

# Standard library
import math

# Third-party
import torch


def compute_grid_weights(coords, grid_type="latlon"):
    """
    Compute area-based grid weights from node coordinates.

    for lat-lon grids weights are cos(latitude). for quasi-uniform
    grids (icosahedral etc) all weights are equal.

    Parameters
    ----------
    coords : torch.Tensor
        node coordinates, shape (N, 2). for latlon grids
        coords[:, 0] = lon, coords[:, 1] = lat (degrees).
    grid_type : str, optional
        ``"latlon"`` or ``"equal_area"``.

    Returns
    -------
    weights : torch.Tensor
        normalised area weights, shape (N,). sums to N so that
        unweighted mean == weighted mean for uniform fields.
    """
    if grid_type == "equal_area":
        return torch.ones(
            coords.shape[0], dtype=coords.dtype, device=coords.device
        )

    if grid_type == "latlon":
        lat_rad = coords[:, 1] * (math.pi / 180.0)
        w = torch.cos(lat_rad).clamp(min=1e-8)
        # normalise so weights sum to N
        w = w * (coords.shape[0] / w.sum())
        return w

    raise ValueError(
        f"Unknown grid_type '{grid_type}'. "
        "Expected 'latlon' or 'equal_area'."
    )


def _apply_weights_and_reduce(
    entry_vals,
    grid_weights=None,
    mask=None,
    reduce_grid=True,
    reduce_vars=False,
):
    """
    apply grid weights and reduce over spatial/variable dims.

    Parameters
    ----------
    entry_vals : torch.Tensor
        per-node values, shape (..., N, d_state).
    grid_weights : torch.Tensor or None
        area weights, shape (N,).
    mask : torch.Tensor or None
        boolean mask for grid nodes, shape (N,).
    reduce_grid : bool
        whether to average over grid dim.
    reduce_vars : bool
        whether to average over variable dim.

    Returns
    -------
    torch.Tensor
        reduced values.
    """
    if mask is not None:
        entry_vals = entry_vals[..., mask, :]
        if grid_weights is not None:
            grid_weights = grid_weights[mask]

    if reduce_grid:
        if grid_weights is not None:
            w = grid_weights.unsqueeze(-1)  # (N, 1)
            entry_vals = (entry_vals * w).sum(dim=-2) / w.sum()
        else:
            entry_vals = torch.mean(entry_vals, dim=-2)

    if reduce_vars:
        entry_vals = torch.mean(entry_vals, dim=-1)

    return entry_vals


def weighted_rmse(
    pred, target, grid_weights=None, mask=None, per_variable=True
):
    """
    Area-weighted RMSE.

    standard metric used by ECMWF etc, weighting each grid point by
    cell area so poles dont dominate on regular latlon grids.

    Parameters
    ----------
    pred : torch.Tensor
        predictions, shape (..., N, d_state).
    target : torch.Tensor
        targets, shape (..., N, d_state).
    grid_weights : torch.Tensor or None
        area weights, shape (N,).
    mask : torch.Tensor or None
        boolean mask, shape (N,).
    per_variable : bool
        if True return per-variable RMSE.

    Returns
    -------
    torch.Tensor
        (..., d_state) if per_variable else (...,).
    """
    sq_err = (pred - target) ** 2  # (..., N, d_state)
    mse = _apply_weights_and_reduce(
        sq_err,
        grid_weights=grid_weights,
        mask=mask,
        reduce_grid=True,
        reduce_vars=(not per_variable),
    )
    return torch.sqrt(mse)


def weighted_mae(pred, target, grid_weights=None, mask=None, per_variable=True):
    """
    Area-weighted MAE.

    Parameters
    ----------
    pred : torch.Tensor
        predictions, shape (..., N, d_state).
    target : torch.Tensor
        targets, shape (..., N, d_state).
    grid_weights : torch.Tensor or None
        area weights, shape (N,).
    mask : torch.Tensor or None
        boolean mask, shape (N,).
    per_variable : bool
        if True return per-variable MAE.

    Returns
    -------
    torch.Tensor
        (..., d_state) if per_variable else (...,).
    """
    abs_err = torch.abs(pred - target)
    return _apply_weights_and_reduce(
        abs_err,
        grid_weights=grid_weights,
        mask=mask,
        reduce_grid=True,
        reduce_vars=(not per_variable),
    )


def latitude_weighted_rmse(pred, target, coords, mask=None, per_variable=True):
    """
    Latitude-weighted RMSE, the standard metric in WeatherBench
    and most NWP intercomparison studies.

    convenience wrapper that computes cos(lat) weights from coords
    and delegates to weighted_rmse.

    Parameters
    ----------
    pred : torch.Tensor
        predictions, shape (..., N, d_state).
    target : torch.Tensor
        targets, shape (..., N, d_state).
    coords : torch.Tensor
        (lon, lat) in degrees, shape (N, 2).
    mask : torch.Tensor or None
        boolean mask, shape (N,).
    per_variable : bool
        if True return per-variable RMSE.

    Returns
    -------
    torch.Tensor
        latitude-weighted RMSE.
    """
    grid_weights = compute_grid_weights(coords, grid_type="latlon")
    return weighted_rmse(
        pred,
        target,
        grid_weights=grid_weights,
        mask=mask,
        per_variable=per_variable,
    )


def acc(pred, target, climatology, grid_weights=None, mask=None):
    """
    Anomaly Correlation Coefficient.

    primary WMO skill score — measures correlation between predicted
    and observed anomalies relative to climatology.
    ACC=1 is perfect, ACC=0 means no skill vs climatology.

    .. math::

        \\text{ACC} = \\frac{
            \\sum_i w_i (f_i - c_i)(o_i - c_i)
        }{
            \\sqrt{\\sum_i w_i (f_i - c_i)^2
                   \\sum_i w_i (o_i - c_i)^2}
        }

    Parameters
    ----------
    pred : torch.Tensor
        forecasts, shape (..., N, d_state).
    target : torch.Tensor
        observations/analysis, shape (..., N, d_state).
    climatology : torch.Tensor
        climatological mean, shape (N, d_state) or broadcastable.
        typically the time-mean of the training set.
    grid_weights : torch.Tensor or None
        area weights, shape (N,).
    mask : torch.Tensor or None
        boolean mask, shape (N,).

    Returns
    -------
    torch.Tensor
        ACC per variable, shape (..., d_state). range [-1, 1].
    """
    pred_anom = pred - climatology
    target_anom = target - climatology

    if mask is not None:
        pred_anom = pred_anom[..., mask, :]
        target_anom = target_anom[..., mask, :]
        if grid_weights is not None:
            grid_weights = grid_weights[mask]

    if grid_weights is not None:
        w = grid_weights.unsqueeze(-1)
    else:
        w = torch.ones(
            pred_anom.shape[-2],
            1,
            dtype=pred_anom.dtype,
            device=pred_anom.device,
        )

    numerator = (w * pred_anom * target_anom).sum(dim=-2)
    pred_var = (w * pred_anom**2).sum(dim=-2)
    target_var = (w * target_anom**2).sum(dim=-2)

    denom = torch.sqrt(pred_var * target_var).clamp(min=1e-12)
    return numerator / denom


def spread_skill_ratio(pred, target, pred_std, grid_weights=None, mask=None):
    """
    Spread-skill ratio for probabilistic calibration.

    well-calibrated model should give ratio ~1.0.
    ratio > 1 means overdispersive, < 1 means overconfident.

    .. math::

        \\text{SSR} = \\frac{\\sqrt{\\text{mean}(\\sigma^2)}}
                           {\\sqrt{\\text{mean}((f-o)^2)}}

    Parameters
    ----------
    pred : torch.Tensor
        predictions, shape (..., N, d_state).
    target : torch.Tensor
        targets, shape (..., N, d_state).
    pred_std : torch.Tensor
        predicted std dev, shape (..., N, d_state) or (d_state,).
    grid_weights : torch.Tensor or None
        area weights, shape (N,).
    mask : torch.Tensor or None
        boolean mask, shape (N,).

    Returns
    -------
    torch.Tensor
        spread-skill ratio per variable, shape (..., d_state).
    """
    # spread: area-weighted mean variance
    mean_var = _apply_weights_and_reduce(
        pred_std**2,
        grid_weights=grid_weights,
        mask=mask,
        reduce_grid=True,
        reduce_vars=False,
    )

    # skill: area-weighted mse
    sq_err = (pred - target) ** 2
    mean_sq_err = _apply_weights_and_reduce(
        sq_err,
        grid_weights=grid_weights,
        mask=mask,
        reduce_grid=True,
        reduce_vars=False,
    )

    spread = torch.sqrt(mean_var.clamp(min=1e-12))
    skill = torch.sqrt(mean_sq_err.clamp(min=1e-12))

    return spread / skill
