# Third-party
import numpy as np
import torch
from cartopy import crs as ccrs


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


def compute_area_weights(datastore):
    """
    Compute area weights for each grid point based on the cosine of the
    latitude. This accounts for the fact that grid cells at higher latitudes
    cover less surface area than cells near the equator.

    The weights are normalized so that they sum to the number of grid points
    (i.e. the mean weight is 1.0), so that using these weights is equivalent
    to an unweighted mean when all latitudes are equal.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore providing grid coordinates and projection information.

    Returns
    -------
    torch.Tensor
        Area weights of shape (N,), where N is the number of grid points.
        Each weight is proportional to cos(latitude) and the weights are
        normalized to sum to N.
    """
    # Get projected x, y coordinates for state grid points
    xy = datastore.get_xy(category="state", stacked=True)  # (N, 2)

    # Transform projected coordinates to geographic lat/lon
    lonlat = ccrs.PlateCarree().transform_points(
        src_crs=datastore.coords_projection,
        x=xy[:, 0],
        y=xy[:, 1],
    )  # (N, 3) -> [lon, lat, z]

    lat_rad = np.deg2rad(lonlat[:, 1])  # latitude in radians
    cos_lat = np.cos(lat_rad)

    # Normalize weights so they sum to N (mean weight = 1.0)
    n_points = len(cos_lat)
    weights = cos_lat * (n_points / cos_lat.sum())

    return torch.tensor(weights, dtype=torch.float32)


def mask_and_reduce_metric(
    metric_entry_vals, mask, average_grid, sum_vars, grid_weights=None
):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced
        (mean over N). If grid_weights are provided, uses weighted mean.
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)
    grid_weights: (N,) or None, optional area-based weights for each grid
        point. When provided and average_grid=True, a weighted mean is used
        instead of a simple mean. The weights are applied after masking.

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[
            ..., mask, :
        ]  # (..., N', d_state)
        # Also mask the grid weights if provided
        if grid_weights is not None:
            grid_weights = grid_weights[mask]  # (N',)

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        if grid_weights is not None:
            # Weighted mean over grid dimension
            # Reshape weights for broadcasting: (N', 1)
            w = grid_weights.to(
                device=metric_entry_vals.device,
                dtype=metric_entry_vals.dtype,
            ).unsqueeze(
                -1
            )  # (N', 1)
            # Normalize weights to sum to 1 over the (possibly masked) grid
            w_sum = w.sum(dim=-2, keepdim=True)
            if w_sum.item() <= 0:
                raise ValueError(
                    "Sum of grid weights is non-positive after masking. "
                    "All weighted grid points may have been masked out."
                )
            w = w / w_sum
            metric_entry_vals = torch.sum(
                metric_entry_vals * w, dim=-2
            )  # (..., d_state)
        else:
            metric_entry_vals = torch.mean(
                metric_entry_vals, dim=-2
            )  # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., N) or (...,)

    return metric_entry_vals


def wmse(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
):
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

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
        grid_weights=grid_weights,
    )


def mse(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
):
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred,
        target,
        torch.ones_like(pred_std),
        mask,
        average_grid,
        sum_vars,
        grid_weights=grid_weights,
    )


def wmae(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
):
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

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
        grid_weights=grid_weights,
    )


def mae(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
):
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred,
        target,
        torch.ones_like(pred_std),
        mask,
        average_grid,
        sum_vars,
        grid_weights=grid_weights,
    )


def nll(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
):
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
        grid_weights=grid_weights,
    )


def crps_gauss(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_weights=None,
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
    grid_weights: (N,) or None, optional area-based weights for each grid
        point

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
        entry_crps,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
        grid_weights=grid_weights,
    )


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
}
