# Third-party
import torch

# First-party
from neural_lam.metrics import (
    crps_gauss,
    mae,
    mask_and_reduce_metric,
    mse,
    nll,
    wmae,
    wmse,
)

# Common test dimensions
N_GRID = 10  # number of grid nodes
D_STATE = 3  # number of state variables
BATCH = 2


def _make_test_data():
    """Create common test tensors."""
    torch.manual_seed(42)
    pred = torch.randn(BATCH, N_GRID, D_STATE)
    target = torch.randn(BATCH, N_GRID, D_STATE)
    pred_std = torch.ones(BATCH, N_GRID, D_STATE)
    return pred, target, pred_std


class TestMaskAndReduceMetric:
    """Tests for grid_weights in mask_and_reduce_metric."""

    def test_no_weights_unchanged(self):
        """grid_weights=None should give identical result to before."""
        vals = torch.randn(BATCH, N_GRID, D_STATE)
        result_none = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=None,
        )
        result_mean = torch.mean(vals, dim=-2)
        assert torch.allclose(result_none, result_mean)

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce same result as no weights."""
        vals = torch.randn(BATCH, N_GRID, D_STATE)
        uniform_weights = torch.ones(N_GRID)
        result_none = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=None,
        )
        result_uniform = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=uniform_weights,
        )
        assert torch.allclose(result_none, result_uniform)

    def test_weights_change_result(self):
        """Non-uniform weights should produce different result."""
        vals = torch.randn(BATCH, N_GRID, D_STATE)
        # Give first node 10x weight
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=None,
        )
        result_weighted = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        assert not torch.allclose(result_none, result_weighted)

    def test_weighted_mean_correctness(self):
        """Verify weighted mean matches manual computation."""
        vals = torch.ones(1, 4, 1)  # (1, 4, 1) - all ones
        vals[0, 0, 0] = 2.0  # first node = 2, rest = 1
        weights = torch.tensor([3.0, 1.0, 1.0, 1.0])
        # Weighted mean = (2*3 + 1*1 + 1*1 + 1*1) / (3+1+1+1)
        #               = (6 + 3) / 6 = 9/6 = 1.5
        expected = torch.tensor([[[1.5]]])
        result = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        assert torch.allclose(result, expected)

    def test_cosine_latitude_weighting(self):
        """Cos-latitude weights should down-weight polar contributions."""
        # Standard library
        import math

        # 4 nodes at different latitudes
        lats_deg = torch.tensor([0.0, 30.0, 60.0, 90.0])
        cos_weights = torch.cos(lats_deg * math.pi / 180.0)

        # Error only at the pole (which has near-zero cos-lat weight)
        vals = torch.zeros(1, 4, 1)
        vals[0, 3, 0] = 1.0  # pole has error 1, rest are 0

        result_weighted = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=cos_weights,
        )
        result_unweighted = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=None,
        )
        # Weighted result should be LOWER because polar error (weight≈0)
        # is down-weighted compared to unweighted mean
        assert result_weighted < result_unweighted

    def test_weights_with_mask(self):
        """Weights should be correctly subsetted when mask is applied."""
        vals = torch.ones(1, 4, 1)
        vals[0, 0, 0] = 10.0
        weights = torch.tensor([5.0, 1.0, 1.0, 1.0])
        mask = torch.tensor([False, True, True, True])

        result = mask_and_reduce_metric(
            vals,
            mask=mask,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        # After masking: vals = [1, 1, 1], weights = [1, 1, 1]
        # Weighted mean = 1.0
        expected = torch.tensor([[[1.0]]])
        assert torch.allclose(result, expected)

    def test_weights_no_average_grid(self):
        """When average_grid=False, weights should have no effect."""
        vals = torch.randn(BATCH, N_GRID, D_STATE)
        weights = torch.randn(N_GRID).abs() + 0.1
        result_none = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=False,
            sum_vars=False,
            grid_weights=None,
        )
        result_weighted = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=False,
            sum_vars=False,
            grid_weights=weights,
        )
        assert torch.allclose(result_none, result_weighted)

    def test_output_shape(self):
        """Weighted reduction should produce same shape as unweighted."""
        vals = torch.randn(BATCH, N_GRID, D_STATE)
        weights = torch.randn(N_GRID).abs() + 0.1
        result = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=True,
            grid_weights=weights,
        )
        assert result.shape == (BATCH,)

        result_no_sum = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        assert result_no_sum.shape == (BATCH, D_STATE)

    def test_zero_weight_sum_raises(self):
        """Weights that sum to zero after masking should raise ValueError."""
        # Third-party
        import pytest

        vals = torch.randn(1, 4, 1)
        # All weights are on nodes that get masked out
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mask = torch.tensor([False, True, True, True])
        with pytest.raises(ValueError, match="positive sum"):
            mask_and_reduce_metric(
                vals,
                mask=mask,
                average_grid=True,
                sum_vars=False,
                grid_weights=weights,
            )

    def test_dtype_casting(self):
        """Float64 weights with float32 values should work via casting."""
        vals = torch.randn(BATCH, N_GRID, D_STATE, dtype=torch.float32)
        weights = torch.ones(N_GRID, dtype=torch.float64)
        # Should not raise a dtype mismatch error
        result = mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        assert result.dtype == torch.float32


class TestMetricFunctionsWithWeights:
    """Verify all 6 metric functions correctly pass grid_weights through."""

    def test_mse_with_weights(self):
        """MSE should support grid_weights and produce different result."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = mse(pred, target, pred_std)
        result_weighted = mse(pred, target, pred_std, grid_weights=weights)
        assert not torch.allclose(result_none, result_weighted)

    def test_wmse_with_weights(self):
        """wMSE should support grid_weights."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = wmse(pred, target, pred_std)
        result_weighted = wmse(pred, target, pred_std, grid_weights=weights)
        assert not torch.allclose(result_none, result_weighted)

    def test_mae_with_weights(self):
        """MAE should support grid_weights."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = mae(pred, target, pred_std)
        result_weighted = mae(pred, target, pred_std, grid_weights=weights)
        assert not torch.allclose(result_none, result_weighted)

    def test_wmae_with_weights(self):
        """wMAE should support grid_weights."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = wmae(pred, target, pred_std)
        result_weighted = wmae(pred, target, pred_std, grid_weights=weights)
        assert not torch.allclose(result_none, result_weighted)

    def test_nll_with_weights(self):
        """NLL should support grid_weights."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = nll(pred, target, pred_std)
        result_weighted = nll(pred, target, pred_std, grid_weights=weights)
        assert not torch.allclose(result_none, result_weighted)

    def test_crps_gauss_with_weights(self):
        """CRPS should support grid_weights."""
        pred, target, pred_std = _make_test_data()
        weights = torch.ones(N_GRID)
        weights[0] = 10.0
        result_none = crps_gauss(pred, target, pred_std)
        result_weighted = crps_gauss(
            pred, target, pred_std, grid_weights=weights
        )
        assert not torch.allclose(result_none, result_weighted)

    def test_all_metrics_uniform_weights_match_none(self):
        """All metrics with uniform weights should match grid_weights=None."""
        pred, target, pred_std = _make_test_data()
        uniform_weights = torch.ones(N_GRID)
        for metric_fn in [mse, wmse, mae, wmae, nll, crps_gauss]:
            result_none = metric_fn(pred, target, pred_std)
            result_uniform = metric_fn(
                pred, target, pred_std, grid_weights=uniform_weights
            )
            assert torch.allclose(result_none, result_uniform, atol=1e-6), (
                f"{metric_fn.__name__} gives different results with "
                f"uniform weights vs no weights"
            )
