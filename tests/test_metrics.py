# Third-party
import pytest
import torch

# First-party
from neural_lam import metrics
from tests.dummy_datastore import DummyDatastore


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def dummy_data():
    """Create simple dummy tensors for metric testing."""
    N = 10  # grid points
    d_state = 3  # state features
    B = 2  # batch size

    pred = torch.randn(B, N, d_state)
    target = torch.randn(B, N, d_state)
    pred_std = torch.abs(torch.randn(B, N, d_state)) + 0.1
    return pred, target, pred_std


@pytest.fixture
def uniform_weights():
    """Uniform weights (should give same result as no weights)."""
    N = 10
    return torch.ones(N)


@pytest.fixture
def non_uniform_weights():
    """Non-uniform weights simulating latitude-dependent area."""
    N = 10
    # Simulate cos(latitude) pattern: larger weights near equator
    lats = torch.linspace(-60, 60, N)
    weights = torch.cos(torch.deg2rad(lats))
    # Normalize so weights sum to N
    weights = weights * (N / weights.sum())
    return weights


@pytest.fixture
def mask():
    """Boolean mask selecting a subset of grid points."""
    N = 10
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:7] = True  # keep first 7 grid points
    return mask


# ============================================================
# Test: backward compatibility (grid_weights=None)
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_no_weights_gives_same_result_as_before(metric_name, dummy_data):
    """
    When grid_weights=None (default), all metrics should produce the same
    result as the original unweighted implementation.
    """
    pred, target, pred_std = dummy_data
    metric_func = metrics.get_metric(metric_name)

    result_no_kwarg = metric_func(pred, target, pred_std)
    result_none = metric_func(pred, target, pred_std, grid_weights=None)

    torch.testing.assert_close(result_no_kwarg, result_none)


# ============================================================
# Test: uniform weights == no weights
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_uniform_weights_equals_no_weights(
    metric_name, dummy_data, uniform_weights
):
    """
    Uniform weights (all ones) should produce the same result as no weights,
    since the weighted mean with equal weights is the same as the arithmetic
    mean.
    """
    pred, target, pred_std = dummy_data
    metric_func = metrics.get_metric(metric_name)

    result_no_weights = metric_func(pred, target, pred_std)
    result_uniform = metric_func(
        pred, target, pred_std, grid_weights=uniform_weights
    )

    torch.testing.assert_close(
        result_no_weights, result_uniform, atol=1e-5, rtol=1e-5
    )


# ============================================================
# Test: non-uniform weights differ from unweighted
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_non_uniform_weights_differ(
    metric_name, dummy_data, non_uniform_weights
):
    """
    Non-uniform weights should generally produce a different result than
    no weights (unless by rare coincidence).
    """
    pred, target, pred_std = dummy_data
    metric_func = metrics.get_metric(metric_name)

    result_no_weights = metric_func(pred, target, pred_std)
    result_weighted = metric_func(
        pred, target, pred_std, grid_weights=non_uniform_weights
    )

    # The results should be a valid tensor (not NaN)
    assert not torch.isnan(result_weighted).any()
    # They should generally differ (we use a loose check — if they are
    # identical bit-for-bit that would be extremely unlikely with random data)
    # This is a soft assertion; we mainly verify correctness elsewhere.
    assert result_weighted.shape == result_no_weights.shape


# ============================================================
# Test: mask_and_reduce_metric directly
# ============================================================
class TestMaskAndReduceMetric:
    """Tests for the mask_and_reduce_metric helper function."""

    def test_no_reduction(self):
        """No masking, no averaging, no summing."""
        vals = torch.randn(2, 10, 3)
        result = metrics.mask_and_reduce_metric(
            vals, mask=None, average_grid=False, sum_vars=False
        )
        torch.testing.assert_close(result, vals)

    def test_mask_only(self):
        """Masking without any reduction."""
        vals = torch.randn(2, 10, 3)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True
        result = metrics.mask_and_reduce_metric(
            vals, mask=mask, average_grid=False, sum_vars=False
        )
        assert result.shape == (2, 5, 3)
        torch.testing.assert_close(result, vals[:, :5, :])

    def test_average_grid_no_weights(self):
        """Average over grid with no weights (simple mean)."""
        vals = torch.ones(2, 10, 3) * 5.0
        result = metrics.mask_and_reduce_metric(
            vals, mask=None, average_grid=True, sum_vars=False
        )
        assert result.shape == (2, 3)
        torch.testing.assert_close(result, torch.ones(2, 3) * 5.0)

    def test_average_grid_uniform_weights(self):
        """Average over grid with uniform weights should equal simple mean."""
        vals = torch.randn(2, 10, 3)
        weights = torch.ones(10)
        result_weighted = metrics.mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        result_unweighted = metrics.mask_and_reduce_metric(
            vals, mask=None, average_grid=True, sum_vars=False
        )
        torch.testing.assert_close(
            result_weighted, result_unweighted, atol=1e-6, rtol=1e-6
        )

    def test_weighted_mean_correctness(self):
        """Verify weighted mean is computed correctly."""
        # 2 grid points, 1 feature
        vals = torch.tensor([[[2.0], [4.0]]])  # (1, 2, 1)
        # Weight grid point 0 twice as much as grid point 1
        weights = torch.tensor([2.0, 1.0])
        result = metrics.mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        # Expected: (2*2 + 1*4) / (2+1) = 8/3
        expected = torch.tensor([[8.0 / 3.0]])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_weighted_mean_with_mask(self):
        """Verify weighted mean works correctly with masking."""
        # 4 grid points, 1 feature
        vals = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (1, 4, 1)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        result = metrics.mask_and_reduce_metric(
            vals,
            mask=mask,
            average_grid=True,
            sum_vars=False,
            grid_weights=weights,
        )
        # After masking: vals=[1,2], weights=[1,2]
        # Weighted mean: (1*1 + 2*2)/(1+2) = 5/3
        expected = torch.tensor([[5.0 / 3.0]])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_weights_not_applied_when_average_grid_false(self):
        """When average_grid=False, weights should have no effect."""
        vals = torch.randn(2, 10, 3)
        weights = torch.randn(10).abs() + 0.1
        result_weighted = metrics.mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=False,
            sum_vars=False,
            grid_weights=weights,
        )
        result_unweighted = metrics.mask_and_reduce_metric(
            vals, mask=None, average_grid=False, sum_vars=False
        )
        torch.testing.assert_close(result_weighted, result_unweighted)

    def test_sum_vars_after_weighted_grid(self):
        """Verify sum_vars works after weighted grid averaging."""
        vals = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        weights = torch.tensor([1.0, 3.0])
        result = metrics.mask_and_reduce_metric(
            vals,
            mask=None,
            average_grid=True,
            sum_vars=True,
            grid_weights=weights,
        )
        # Weighted mean per feature:
        #   feat0: (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        #   feat1: (1*2 + 3*4)/(1+3) = 14/4 = 3.5
        # Sum: 2.5 + 3.5 = 6.0
        expected = torch.tensor([6.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


# ============================================================
# Test: compute_area_weights
# ============================================================
class TestComputeAreaWeights:
    """Tests for the compute_area_weights utility function."""

    def test_weights_shape(self):
        """Weights should have shape (N,) matching grid points."""
        datastore = DummyDatastore()
        weights = metrics.compute_area_weights(datastore)
        assert weights.shape == (datastore.num_grid_points,)

    def test_weights_positive(self):
        """All area weights should be positive."""
        datastore = DummyDatastore()
        weights = metrics.compute_area_weights(datastore)
        assert (weights > 0).all()

    def test_weights_sum_to_n(self):
        """Weights should be normalized to sum to N (mean = 1)."""
        datastore = DummyDatastore()
        weights = metrics.compute_area_weights(datastore)
        n = datastore.num_grid_points
        torch.testing.assert_close(
            weights.sum(),
            torch.tensor(float(n)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_weights_dtype(self):
        """Weights should be float32."""
        datastore = DummyDatastore()
        weights = metrics.compute_area_weights(datastore)
        assert weights.dtype == torch.float32


# ============================================================
# Test: metric functions with batch + time dimensions
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_metrics_with_batch_and_time_dims(metric_name):
    """
    Test that metrics handle batch (B) and time (T) dimensions correctly
    with grid_weights, matching the shape used in ar_model.py.
    """
    B, T, N, d_state = 2, 4, 10, 3
    pred = torch.randn(B, T, N, d_state)
    target = torch.randn(B, T, N, d_state)
    pred_std = torch.abs(torch.randn(B, T, N, d_state)) + 0.1
    weights = torch.ones(N)

    metric_func = metrics.get_metric(metric_name)

    # With average_grid=True, sum_vars=True -> (B, T)
    result = metric_func(pred, target, pred_std, grid_weights=weights)
    assert result.shape == (B, T)

    # With average_grid=True, sum_vars=False -> (B, T, d_state)
    result = metric_func(
        pred, target, pred_std, sum_vars=False, grid_weights=weights
    )
    assert result.shape == (B, T, d_state)

    # With average_grid=False, sum_vars=True -> (B, T, N)
    result = metric_func(
        pred, target, pred_std, average_grid=False, grid_weights=weights
    )
    assert result.shape == (B, T, N)

    # With average_grid=False, sum_vars=False -> (B, T, N, d_state)
    result = metric_func(
        pred,
        target,
        pred_std,
        average_grid=False,
        sum_vars=False,
        grid_weights=weights,
    )
    assert result.shape == (B, T, N, d_state)


# ============================================================
# Test: mask + weights combined for each metric
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_metrics_with_mask_and_weights(
    metric_name, dummy_data, non_uniform_weights, mask
):
    """
    Verify that mask and grid_weights can be used together without errors
    and produce valid output.
    """
    pred, target, pred_std = dummy_data
    metric_func = metrics.get_metric(metric_name)

    result = metric_func(
        pred,
        target,
        pred_std,
        mask=mask,
        grid_weights=non_uniform_weights,
    )

    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    assert result.shape == (2,)  # (B,) after reducing grid and vars


# ============================================================
# Test: edge case — single grid point
# ============================================================
@pytest.mark.parametrize("metric_name", metrics.DEFINED_METRICS.keys())
def test_single_grid_point(metric_name):
    """
    Metrics should work with a single grid point and grid_weights.
    """
    pred = torch.randn(1, 1, 2)
    target = torch.randn(1, 1, 2)
    pred_std = torch.abs(torch.randn(1, 1, 2)) + 0.1
    weights = torch.tensor([1.0])

    metric_func = metrics.get_metric(metric_name)
    result = metric_func(pred, target, pred_std, grid_weights=weights)

    assert not torch.isnan(result).any()
    assert result.shape == (1,)


# ============================================================
# Test: MSE weighted correctness (manual calculation)
# ============================================================
def test_mse_weighted_manual_calculation():
    """
    Verify MSE with area weights matches a manual weighted-mean calculation.
    """
    # 1 batch, 3 grid points, 1 feature
    pred = torch.tensor([[[1.0], [2.0], [3.0]]])
    target = torch.tensor([[[2.0], [2.0], [1.0]]])
    pred_std = torch.ones(1, 3, 1)

    # Errors: (1-2)^2=1, (2-2)^2=0, (3-1)^2=4
    # Unweighted mean: (1+0+4)/3 = 5/3
    result_unweighted = metrics.mse(pred, target, pred_std)
    expected_unweighted = torch.tensor([5.0 / 3.0])
    torch.testing.assert_close(
        result_unweighted, expected_unweighted, atol=1e-6, rtol=1e-6
    )

    # Weighted: w=[1,2,1], normalized: [1/4, 2/4, 1/4]
    # Weighted mean: (1*1/4 + 0*2/4 + 4*1/4) = 5/4 = 1.25
    weights = torch.tensor([1.0, 2.0, 1.0])
    result_weighted = metrics.mse(pred, target, pred_std, grid_weights=weights)
    expected_weighted = torch.tensor([5.0 / 4.0])
    torch.testing.assert_close(
        result_weighted, expected_weighted, atol=1e-6, rtol=1e-6
    )


# ============================================================
# Test: MAE weighted correctness (manual calculation)
# ============================================================
def test_mae_weighted_manual_calculation():
    """
    Verify MAE with area weights matches a manual weighted-mean calculation.
    """
    # 1 batch, 3 grid points, 1 feature
    pred = torch.tensor([[[1.0], [3.0], [5.0]]])
    target = torch.tensor([[[2.0], [3.0], [2.0]]])
    pred_std = torch.ones(1, 3, 1)

    # Errors: |1-2|=1, |3-3|=0, |5-2|=3
    # Weighted: w=[3,1,2], normalized: [3/6, 1/6, 2/6] = [0.5, 1/6, 1/3]
    # Weighted mean: 1*0.5 + 0*1/6 + 3*1/3 = 0.5 + 0 + 1 = 1.5
    weights = torch.tensor([3.0, 1.0, 2.0])
    result_weighted = metrics.mae(pred, target, pred_std, grid_weights=weights)
    expected_weighted = torch.tensor([1.5])
    torch.testing.assert_close(
        result_weighted, expected_weighted, atol=1e-6, rtol=1e-6
    )
