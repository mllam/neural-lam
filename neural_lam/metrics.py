# Third-party
import torch
import torch.fft


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


def wmse(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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


def mse(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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


def mae(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, **kwargs
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


def log_spectral_distance(
    pred,
    target,
    pred_std,
    mask=None,
    average_grid=True,
    sum_vars=True,
    grid_shape=None,
    edge_index=None,
    num_moments=10,
    eps=1e-8,
):
    """
    Log-Spectral Distance (LSD)

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (unused)
    mask: (N,), boolean mask describing which grid nodes to use (unused)
    average_grid: boolean, if result should be averaged over grid
    sum_vars: boolean, if variable dimension -1 should be reduced
    grid_shape: tuple (ny, nx), shape of the 2D grid (for regular grids)
    edge_index: (2, M), edges in the graph (for unstructured grids)
    num_moments: int, number of Laplacian moments to use for unstructured LSD
    eps: float, small value to avoid log(0)

    Returns:
    metric_val: One of (...,), (..., d_state), depends on reduction arguments.
    """
    # Regular grid LSD using FFT
    if grid_shape is not None or (edge_index is None and _is_square_grid(pred)):
        if grid_shape is None:
            num_nodes = pred.shape[-2]
            side = int(num_nodes**0.5)
            grid_shape = (side, side)

        # Reshape to (..., d_state, ny, nx) for FFT
        ny, nx = grid_shape
        # Move d_state to before grid dimensions
        # pred is (..., N, d_state) -> (..., d_state, N) -> (..., d_state, ny, nx)
        p = pred.transpose(-1, -2).reshape(
            *pred.shape[:-2], pred.shape[-1], ny, nx
        )
        t = target.transpose(-1, -2).reshape(
            *target.shape[:-2], target.shape[-1], ny, nx
        )

        # Compute 2D RFFT
        f_pred = torch.fft.rfft2(p, norm="ortho")
        f_target = torch.fft.rfft2(t, norm="ortho")

        # Power Spectrum: |F(u,v)|^2
        ps_pred = torch.abs(f_pred) ** 2
        ps_target = torch.abs(f_target) ** 2

        # Average over frequency dimensions
        # entry_lsd is (..., d_state, freq_y, freq_x)
        # We compute mean( (10 * log10(P_target/P_pred))^2 ) then sqrt
        diff_lsd = (10 * torch.log10((ps_target + eps) / (ps_pred + eps))) ** 2
        metric_val = torch.mean(diff_lsd, dim=(-2, -1))  # (..., d_state)
        metric_val = torch.sqrt(metric_val)

    # Unstructured grid LSD using Graph Signal Processing
    elif edge_index is not None:
        # Compute spectral moments using Normalized Laplacian
        # moments: (..., d_state, num_moments)
        m_pred = _compute_laplacian_moments(pred, edge_index, num_moments)
        m_target = _compute_laplacian_moments(target, edge_index, num_moments)

        # Log-Spectral Distance over moments:
        # RMS of 10 * log10(m_target / m_pred)
        # diff_lsd is (..., d_state, num_moments)
        diff_lsd = (10 * torch.log10((m_target + eps) / (m_pred + eps))) ** 2
        metric_val = torch.mean(diff_lsd, dim=-1)  # (..., d_state)
        metric_val = torch.sqrt(metric_val)

    else:
        raise ValueError(
            "log_spectral_distance requires grid_shape, edge_index, "
            "or a square grid"
        )

    if sum_vars:
        metric_val = torch.sum(metric_val, dim=-1)  # (...,)

    return metric_val


def _is_square_grid(pred):
    """Check if the grid dimension is a perfect square"""
    num_nodes = pred.shape[-2]
    side = int(num_nodes**0.5)
    return side**2 == num_nodes


def _compute_laplacian_moments(x, edge_index, num_moments):
    """
    Compute moments of the spectral distribution: m_k = x^T L^k x
    where L is the Normalized Laplacian.
    """
    # x: (..., N, d_state)
    # edge_index: (2, M)
    # returns: (..., d_state, num_moments)
    N = x.shape[-2]
    device = x.device

    # 1. Compute Normalized Laplacian as a sparse matrix
    # L = I - D^-1/2 A D^-1/2
    row, col = edge_index
    deg = torch.zeros(N, device=device)
    # Assume unweighted adjacency for now, or use edge_weight if provided
    # For neural-lam, m2m_edge_index is usually unweighted or has features
    deg.scatter_add_(0, col, torch.ones_like(row, dtype=torch.float32))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    # Normalized weights: -1 / sqrt(di * dj)
    val = -deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Sparse Laplacian L
    # Off-diagonal: -D^-1/2 A D^-1/2
    indices = torch.cat(
        [edge_index, torch.stack([torch.arange(N, device=device)] * 2)], dim=1
    )
    values = torch.cat([val, torch.ones(N, device=device)])
    L = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    # 2. Iteratively compute x_k = L^k x and m_k = x^T x_k
    # x is (..., N, d_state)
    # Reshape x to (N, -1) for sparse mm
    orig_shape = x.shape
    x_flat = x.transpose(-2, -1).reshape(-1, N).t()  # (N, B * d_state)

    moments = []
    curr_x = x_flat
    for _ in range(num_moments):
        # m_k = x^T (L^k x)
        # dot product per column
        m_k = torch.sum(x_flat * curr_x, dim=0)  # (B * d_state,)
        moments.append(m_k)
        # curr_x = L * curr_x
        curr_x = torch.sparse.mm(L, curr_x)

    # moments: list of (B * d_state,)
    moments = torch.stack(moments, dim=-1)  # (B * d_state, num_moments)

    # Reshape back to (..., d_state, num_moments)
    res = moments.view(*orig_shape[:-2], orig_shape[-1], num_moments)

    # Normalize moments by k=0 (total energy) to get relative distribution?
    # Actually, standard LSD compares absolute power spectra.
    # But if we want it to be scale-invariant, we could.
    # The proposal didn't specify, so we use absolute moments for now.
    # We take absolute value to ensure positivity before log
    return torch.abs(res)


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "lsd": log_spectral_distance,
}
