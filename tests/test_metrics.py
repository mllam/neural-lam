# Third-party
import pytest
import torch

# First-party
from neural_lam.metrics import (
    DEFINED_METRICS,
    crps_gauss,
    get_metric,
    mae,
    mask_and_reduce_metric,
    mse,
    nll,
    wmae,
    wmse,
)


@pytest.mark.parametrize("metric_name", list(DEFINED_METRICS.keys()))
def test_get_metric_valid(metric_name: str) -> None:
    """`get_metric` retrieves the correct function (case-insensitive)."""
    fn_lower = get_metric(metric_name.lower())
    fn_upper = get_metric(metric_name.upper())
    assert fn_lower == DEFINED_METRICS[metric_name.lower()]
    assert fn_upper == DEFINED_METRICS[metric_name.lower()]


@pytest.mark.parametrize("invalid_name", ["unknown", "rmse", "", "invalid_metric"])
def test_get_metric_invalid_raises_value_error(invalid_name: str) -> None:
    """`get_metric` raises ValueError with a descriptive message on invalid names."""
    with pytest.raises(ValueError, match=f"Unknown metric: '{invalid_name}'"):
        get_metric(invalid_name)


def test_mask_and_reduce_metric_shapes_and_values() -> None:
    """`mask_and_reduce_metric` reduces and masks tensor dimensions correctly."""
    # Shape: (B, N, num_vars) = (2, 4, 3)
    vals = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]],
        ]
    )

    # 1. No reduction: (average_grid=False, sum_vars=False) -> (2, 4, 3)
    res_none = mask_and_reduce_metric(
        vals, mask=None, average_grid=False, sum_vars=False
    )
    assert res_none.shape == (2, 4, 3)
    torch.testing.assert_close(res_none, vals)

    # 2. Reduce grid only: average_grid=True, sum_vars=False -> (2, 3)
    res_grid = mask_and_reduce_metric(
        vals, mask=None, average_grid=True, sum_vars=False
    )
    assert res_grid.shape == (2, 3)
    expected_grid = torch.mean(vals, dim=-2)
    torch.testing.assert_close(res_grid, expected_grid)

    # 3. Reduce vars only: average_grid=False, sum_vars=True -> (2, 4)
    res_vars = mask_and_reduce_metric(
        vals, mask=None, average_grid=False, sum_vars=True
    )
    assert res_vars.shape == (2, 4)
    expected_vars = torch.sum(vals, dim=-1)
    torch.testing.assert_close(res_vars, expected_vars)

    # 4. Full reduction: average_grid=True, sum_vars=True -> (2,)
    res_full = mask_and_reduce_metric(
        vals, mask=None, average_grid=True, sum_vars=True
    )
    assert res_full.shape == (2,)
    expected_full = torch.sum(torch.mean(vals, dim=-2), dim=-1)
    torch.testing.assert_close(res_full, expected_full)

    # 5. With boolean mask selecting nodes 0 and 2
    mask = torch.tensor([True, False, True, False])
    res_masked = mask_and_reduce_metric(
        vals, mask=mask, average_grid=True, sum_vars=True
    )
    assert res_masked.shape == (2,)
    expected_masked = torch.sum(torch.mean(vals[:, [0, 2], :], dim=-2), dim=-1)
    torch.testing.assert_close(res_masked, expected_masked)


def test_mse_and_wmse_computation() -> None:
    """`mse` and `wmse` compute correct unweighted and weighted squared errors."""
    pred = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])  # shape (1, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    pred_std = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])

    # Unweighted MSE:
    # diffs = [1, 2], [3, 4] -> diffs^2 = [1, 4], [9, 16]
    # mean over 2 nodes: [(1+9)/2, (4+16)/2] = [5, 10]
    # sum over vars: 5 + 10 = 15
    res_mse = mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)
    assert res_mse.item() == pytest.approx(15.0)

    # Weighted MSE (divided by pred_std^2 = 4):
    # 15 / 4 = 3.75
    res_wmse = wmse(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
    )
    assert res_wmse.item() == pytest.approx(3.75)


def test_mae_and_wmae_computation() -> None:
    """`mae` and `wmae` compute correct unweighted and weighted absolute errors."""
    pred = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])  # shape (1, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    pred_std = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])

    # Unweighted MAE:
    # diffs = [1, 2], [3, 4]
    # mean over 2 nodes: [(1+3)/2, (2+4)/2] = [2, 3]
    # sum over vars: 2 + 3 = 5
    res_mae = mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)
    assert res_mae.item() == pytest.approx(5.0)

    # Weighted MAE (divided by pred_std = 2):
    # 5 / 2 = 2.5
    res_wmae = wmae(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
    )
    assert res_wmae.item() == pytest.approx(2.5)


def test_probabilistic_metrics_computation() -> None:
    """`nll` and `crps_gauss` compute finite and expected outputs."""
    pred = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # shape (1, 2, 2)
    target = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # identical target
    pred_std = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])

    res_nll = nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)
    res_crps = crps_gauss(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
    )

    assert torch.isfinite(res_nll).all()
    assert torch.isfinite(res_crps).all()
