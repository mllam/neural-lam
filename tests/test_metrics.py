# Third-party
import matplotlib.pyplot as plt
import pytest
import torch

# First-party
from neural_lam.metrics import (
    MAE,
    MSE,
    NLL,
    WMAE,
    WMSE,
    BaseMetric,
    CRPSGauss,
    OutputStd,
    get_metric,
    mask_and_reduce_metric,
)


class TestBaseMetricInterface:
    """Test that all metric objects conform to the BaseMetric interface."""

    @pytest.mark.parametrize(
        "metric_name",
        ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss", "output_std"],
    )
    def test_get_metric_returns_base_metric(self, metric_name):
        """get_metric should return a BaseMetric instance."""
        metric = get_metric(metric_name)
        assert isinstance(metric, BaseMetric)

    @pytest.mark.parametrize(
        "metric_name",
        ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss", "output_std"],
    )
    def test_metric_has_required_attributes(self, metric_name):
        """Each metric must have the full logging-semantics interface."""
        metric = get_metric(metric_name)
        assert hasattr(metric, "name")
        assert hasattr(metric, "__name__")
        assert hasattr(metric, "display_name")
        assert hasattr(metric, "aggregate")
        assert hasattr(metric, "post_process")
        assert hasattr(metric, "rescale")
        assert hasattr(metric, "prepare_for_logging")
        assert callable(metric)

    @pytest.mark.parametrize(
        "metric_name",
        ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss", "output_std"],
    )
    def test_metric_exposes_callable_name(self, metric_name):
        """Metric objects should expose a function-like __name__."""
        metric = get_metric(metric_name)
        assert metric.__name__ == metric_name

    def test_unknown_metric_raises(self):
        """get_metric should raise for unknown metric names."""
        with pytest.raises(AssertionError, match="Unknown metric"):
            get_metric("nonexistent_metric")


class TestMetricComputation:
    """Test that metric computations produce correct values."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prediction, target, and pred_std tensors."""
        torch.manual_seed(42)
        pred = torch.randn(2, 3, 4, 5)  # (B, pred_steps, N, d_state)
        target = torch.randn(2, 3, 4, 5)
        pred_std = torch.ones(5)  # (d_state,)
        return pred, target, pred_std

    def test_mse_callable(self, sample_data):
        """MSE metric should be callable and return correct shape."""
        pred, target, pred_std = sample_data
        metric = MSE()
        result = metric(pred, target, pred_std)
        assert result.shape == (2, 3)  # (B, pred_steps)
        assert (result >= 0).all()

    def test_mae_callable(self, sample_data):
        """MAE metric should be callable and return correct shape."""
        pred, target, pred_std = sample_data
        metric = MAE()
        result = metric(pred, target, pred_std)
        assert result.shape == (2, 3)  # (B, pred_steps)
        assert (result >= 0).all()

    def test_mse_equals_manual(self, sample_data):
        """MSE should match manual computation."""
        pred, target, pred_std = sample_data
        metric = MSE()
        # With sum_vars=False, average_grid=True
        result = metric(
            pred, target, pred_std, average_grid=True, sum_vars=False
        )
        expected = torch.mean(
            torch.nn.functional.mse_loss(pred, target, reduction="none"),
            dim=-2,
        )
        torch.testing.assert_close(result, expected)

    def test_mae_equals_manual(self, sample_data):
        """MAE should match manual computation."""
        pred, target, pred_std = sample_data
        metric = MAE()
        result = metric(
            pred, target, pred_std, average_grid=True, sum_vars=False
        )
        expected = torch.mean(
            torch.nn.functional.l1_loss(pred, target, reduction="none"),
            dim=-2,
        )
        torch.testing.assert_close(result, expected)

    def test_wmse_with_unit_std_equals_mse(self, sample_data):
        """WMSE with pred_std=1 should equal MSE."""
        pred, target, pred_std = sample_data
        mse_metric = MSE()
        wmse_metric = WMSE()
        mse_result = mse_metric(pred, target, pred_std)
        wmse_result = wmse_metric(pred, target, pred_std)
        torch.testing.assert_close(mse_result, wmse_result)

    def test_wmae_with_unit_std_equals_mae(self, sample_data):
        """WMAE with pred_std=1 should equal MAE."""
        pred, target, pred_std = sample_data
        mae_metric = MAE()
        wmae_metric = WMAE()
        mae_result = mae_metric(pred, target, pred_std)
        wmae_result = wmae_metric(pred, target, pred_std)
        torch.testing.assert_close(mae_result, wmae_result)

    def test_nll_callable(self, sample_data):
        """NLL metric should be callable and return finite values."""
        pred, target, pred_std = sample_data
        metric = NLL()
        result = metric(pred, target, pred_std)
        assert result.shape == (2, 3)  # (B, pred_steps)
        assert torch.isfinite(result).all()

    def test_crps_gauss_callable(self, sample_data):
        """CRPS Gauss metric should be callable and return correct shape."""
        pred, target, pred_std = sample_data
        metric = CRPSGauss()
        result = metric(pred, target, pred_std)
        assert result.shape == (2, 3)  # (B, pred_steps)

    def test_output_std_callable(self, sample_data):
        """OutputStd should reduce predicted std like any other metric."""
        pred, target, _ = sample_data
        pred_std = torch.full_like(pred, 2.5)
        metric = OutputStd()
        result = metric(
            pred, target, pred_std, average_grid=True, sum_vars=False
        )
        expected = torch.full((2, 3, 5), 2.5)
        torch.testing.assert_close(result, expected)

    def test_mask_reduces_grid(self, sample_data):
        """Masking should reduce the number of grid nodes used."""
        pred, target, pred_std = sample_data
        mask = torch.tensor([True, False, True, False])
        metric = MSE()
        result_masked = metric(
            pred,
            target,
            pred_std,
            mask=mask,
            average_grid=True,
            sum_vars=False,
        )
        result_full = metric(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=True,
            sum_vars=False,
        )
        # Masked result uses fewer grid nodes -- values will differ
        assert result_masked.shape == result_full.shape
        assert not torch.allclose(result_masked, result_full)


class TestPostProcessAndRescale:
    """
    Test the post_process and rescale behavior of each metric.
    This is the core of the #343 fix -- each metric defines its own
    aggregation semantics instead of relying on hardcoded rules.
    """

    @pytest.fixture
    def averaged_tensor(self):
        """Simulated batch-averaged metric tensor."""
        return torch.tensor([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])

    @pytest.fixture
    def state_std(self):
        """Per-variable standard deviations."""
        return torch.tensor([2.0, 3.0, 5.0])

    def test_mse_post_process_applies_sqrt(self, averaged_tensor):
        """MSE post_process should apply sqrt (MSE -> RMSE)."""
        metric = MSE()
        result = metric.post_process(averaged_tensor)
        expected = torch.sqrt(averaged_tensor)
        torch.testing.assert_close(result, expected)

    def test_mse_display_name_is_rmse(self):
        """MSE display_name should be 'rmse'."""
        assert MSE().display_name == "rmse"

    def test_mse_rescale_is_linear(self, averaged_tensor, state_std):
        """MSE rescale should multiply by state_std."""
        metric = MSE()
        result = metric.rescale(averaged_tensor, state_std)
        expected = averaged_tensor * state_std
        torch.testing.assert_close(result, expected)

    def test_mae_post_process_is_identity(self, averaged_tensor):
        """MAE post_process should be identity."""
        metric = MAE()
        result = metric.post_process(averaged_tensor)
        torch.testing.assert_close(result, averaged_tensor)

    def test_mae_display_name(self):
        """MAE display_name should be 'mae'."""
        assert MAE().display_name == "mae"

    def test_mae_rescale_is_linear(self, averaged_tensor, state_std):
        """MAE rescale should multiply by state_std."""
        metric = MAE()
        result = metric.rescale(averaged_tensor, state_std)
        expected = averaged_tensor * state_std
        torch.testing.assert_close(result, expected)

    def test_wmse_post_process_applies_sqrt(self, averaged_tensor):
        """WMSE post_process should apply sqrt (WMSE -> WRMSE)."""
        metric = WMSE()
        result = metric.post_process(averaged_tensor)
        expected = torch.sqrt(averaged_tensor)
        torch.testing.assert_close(result, expected)

    def test_wmse_display_name_is_wrmse(self):
        """WMSE display_name should be 'wrmse'."""
        assert WMSE().display_name == "wrmse"

    def test_wmse_rescale_is_identity(self, averaged_tensor, state_std):
        """WMSE should stay dimensionless when logged."""
        metric = WMSE()
        result = metric.rescale(averaged_tensor, state_std)
        torch.testing.assert_close(result, averaged_tensor)

    def test_wmae_post_process_is_identity(self, averaged_tensor):
        """WMAE post_process should be identity."""
        metric = WMAE()
        result = metric.post_process(averaged_tensor)
        torch.testing.assert_close(result, averaged_tensor)

    def test_wmae_rescale_is_identity(self, averaged_tensor, state_std):
        """WMAE should stay dimensionless when logged."""
        metric = WMAE()
        result = metric.rescale(averaged_tensor, state_std)
        torch.testing.assert_close(result, averaged_tensor)

    def test_nll_post_process_is_identity(self, averaged_tensor):
        """NLL post_process should be identity."""
        metric = NLL()
        result = metric.post_process(averaged_tensor)
        torch.testing.assert_close(result, averaged_tensor)

    def test_nll_rescale_is_identity(self, averaged_tensor, state_std):
        """NLL rescale should NOT rescale (identity)."""
        metric = NLL()
        result = metric.rescale(averaged_tensor, state_std)
        torch.testing.assert_close(result, averaged_tensor)

    def test_nll_display_name(self):
        """NLL display_name should be 'nll'."""
        assert NLL().display_name == "nll"

    def test_crps_gauss_post_process_is_identity(self, averaged_tensor):
        """CRPS Gauss post_process should be identity."""
        metric = CRPSGauss()
        result = metric.post_process(averaged_tensor)
        torch.testing.assert_close(result, averaged_tensor)

    def test_crps_gauss_rescale_is_linear(self, averaged_tensor, state_std):
        """CRPS Gauss rescale should multiply by state_std."""
        metric = CRPSGauss()
        result = metric.rescale(averaged_tensor, state_std)
        expected = averaged_tensor * state_std
        torch.testing.assert_close(result, expected)

    def test_output_std_post_process_is_identity(self, averaged_tensor):
        """OutputStd post_process should be identity."""
        metric = OutputStd()
        result = metric.post_process(averaged_tensor)
        torch.testing.assert_close(result, averaged_tensor)

    def test_output_std_rescale_is_linear(self, averaged_tensor, state_std):
        """OutputStd rescale should multiply by state_std."""
        metric = OutputStd()
        result = metric.rescale(averaged_tensor, state_std)
        expected = averaged_tensor * state_std
        torch.testing.assert_close(result, expected)

    def test_prepare_for_logging_uses_metric_aggregation(self, state_std):
        """The metric object should own the epoch-level aggregation step."""

        class SumMetric(BaseMetric):
            name = "sum_metric"

            def compute_entry_vals(self, pred, target, pred_std):
                return pred

            def aggregate(self, metric_tensor):
                return torch.sum(metric_tensor, dim=0)

        metric = SumMetric()
        metric_tensor = torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
            ]
        )
        result = metric.prepare_for_logging(metric_tensor, state_std)
        expected = torch.tensor([[10.0, 21.0, 45.0]])
        torch.testing.assert_close(result, expected)


class TestMaskAndReduceMetric:
    """Test the mask_and_reduce_metric helper function."""

    def test_no_reduction(self):
        """Without reduction, should return masked values."""
        vals = torch.randn(2, 3, 4, 5)
        result = mask_and_reduce_metric(
            vals, mask=None, average_grid=False, sum_vars=False
        )
        torch.testing.assert_close(result, vals)

    def test_average_grid_only(self):
        """Should average over grid dimension (-2) only."""
        vals = torch.randn(2, 3, 4, 5)
        result = mask_and_reduce_metric(
            vals, mask=None, average_grid=True, sum_vars=False
        )
        assert result.shape == (2, 3, 5)

    def test_sum_vars_only(self):
        """Should sum over variable dimension (-1) only."""
        vals = torch.randn(2, 3, 4, 5)
        result = mask_and_reduce_metric(
            vals, mask=None, average_grid=False, sum_vars=True
        )
        assert result.shape == (2, 3, 4)

    def test_both_reductions(self):
        """Should reduce both grid and variable dimensions."""
        vals = torch.randn(2, 3, 4, 5)
        result = mask_and_reduce_metric(
            vals, mask=None, average_grid=True, sum_vars=True
        )
        assert result.shape == (2, 3)

    def test_mask_applied(self):
        """Boolean mask should filter grid nodes."""
        vals = torch.ones(2, 4, 5)  # (B, N, d_state)
        mask = torch.tensor([True, False, True, False])
        result = mask_and_reduce_metric(
            vals, mask=mask, average_grid=False, sum_vars=False
        )
        assert result.shape == (2, 2, 5)  # 2 nodes kept


class TestBackwardCompatibility:
    """
    Test that the metric objects maintain backward compatibility with the
    function-based API. The get_metric() call should return a callable that
    matches the old function signature.
    """

    @pytest.fixture
    def sample_data(self):
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 5)  # (B, N, d_state)
        target = torch.randn(2, 4, 5)
        pred_std = torch.ones(5)
        return pred, target, pred_std

    @pytest.mark.parametrize(
        "metric_name", ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss"]
    )
    def test_metric_callable_with_old_signature(self, metric_name, sample_data):
        """Metric objects should work with the same call signature as the
        old functions."""
        pred, target, pred_std = sample_data
        metric = get_metric(metric_name)
        # Old-style call: metric_func(pred, target, pred_std, ...)
        result = metric(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=True,
            sum_vars=True,
        )
        assert result.shape == (2,)

    @pytest.mark.parametrize(
        "metric_name", ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss"]
    )
    def test_metric_with_sum_vars_false(self, metric_name, sample_data):
        """Metric objects with sum_vars=False should return per-variable
        values."""
        pred, target, pred_std = sample_data
        metric = get_metric(metric_name)
        result = metric(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=True,
            sum_vars=False,
        )
        assert result.shape == (2, 5)  # (B, d_state)


def test_aggregate_and_plot_metrics_supports_unknown_metric_keys():
    """Unknown metric keys should still use the generic aggregation path."""

    class MockLogger:
        def log_image(self, key, images):
            pass

    class MockModule:
        def __init__(self):
            self.state_std = torch.tensor([2.0, 3.0])
            self.trainer = type(
                "Trainer",
                (),
                {
                    "is_global_zero": True,
                    "sanity_checking": False,
                    "current_epoch": 0,
                },
            )()
            self.logger = MockLogger()
            self.captured = None

        def all_gather_cat(self, tensor):
            return tensor

        def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
            self.captured = (metric_tensor, prefix, metric_name)
            return {f"{prefix}_{metric_name}": plt.figure()}

    module = MockModule()
    # First-party
    from neural_lam.models.ar_model import ARModel

    module.aggregate_and_plot_metrics = (
        ARModel.aggregate_and_plot_metrics.__get__(module, MockModule)
    )

    custom_metric_vals = [torch.ones(2, 3, 2)]
    module.aggregate_and_plot_metrics(
        {"custom_metric": custom_metric_vals}, prefix="test"
    )

    metric_tensor, prefix, metric_name = module.captured
    torch.testing.assert_close(
        metric_tensor, torch.tensor([[2.0, 3.0]]).repeat(3, 1)
    )
    assert prefix == "test"
    assert metric_name == "custom_metric"
