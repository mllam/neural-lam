# Third-party
import pytest
import torch

# First-party
from neural_lam.metrics import (
    crps_gauss,
    get_metric,
    mae,
    mask_and_reduce_metric,
    mse,
    nll,
    wmae,
    wmse,
)


class TestGetMetric:
    """Tests for metric lookup function."""

    def test_valid_metric_names(self):
        for name in ["mse", "mae", "wmse", "wmae", "nll", "crps_gauss"]:
            assert callable(get_metric(name))

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("unknown_metric")

    def test_metric_name_case_insensitive(self):
        assert get_metric("MSE") is get_metric("mse")


class TestMSE:
    """Known-value tests for MSE."""

    def test_perfect_prediction(self):
        pred = target = torch.randn(2, 10, 5)
        result = mse(pred, target, torch.ones(5))
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self):
        pred = torch.tensor([[[1.0, 2.0]]])
        target = torch.tensor([[[2.0, 4.0]]])
        std = torch.ones(2)
        # MSE = (1 + 4) / 1 grid point = 5.0 (sum_vars=True)
        result = mse(pred, target, std, average_grid=True, sum_vars=True)
        assert torch.allclose(result, torch.tensor(5.0))


class TestMaskBehavior:
    """Tests that masking correctly excludes grid points."""

    def test_mask_excludes_points(self):
        pred = torch.zeros(1, 4, 2)
        target = torch.ones(1, 4, 2)
        mask = torch.tensor([True, True, False, False])
        result = mse(pred, target, torch.ones(2), mask=mask)
        assert torch.allclose(result, torch.tensor(2.0))

    def test_mask_and_reduce_metric_direct(self):
        # (2, 4, 3): 2 batch, 4 grid, 3 vars
        entry_vals = torch.ones(2, 4, 3)
        mask = torch.tensor([True, False, True, False])
        result = mask_and_reduce_metric(
            entry_vals, mask=mask, average_grid=True, sum_vars=True
        )
        assert result.shape == (2,)
        assert torch.allclose(result, torch.ones(2) * 3.0)  # mean of 1s = 1, sum vars = 3

    def test_mask_none_includes_all(self):
        pred = torch.zeros(1, 4, 2)
        target = torch.ones(1, 4, 2)
        result_no_mask = mse(pred, target, torch.ones(2), mask=None)
        result_full_mask = mse(
            pred, target, torch.ones(2), mask=torch.ones(4, dtype=torch.bool)
        )
        assert torch.allclose(result_no_mask, result_full_mask)


class TestReductionModes:
    """Test all 4 combinations of average_grid x sum_vars."""

    @pytest.mark.parametrize(
        "avg_grid,sum_v",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_output_shape(self, avg_grid, sum_v):
        B, N, D = 3, 10, 5
        pred = torch.randn(B, N, D)
        target = torch.randn(B, N, D)
        result = mse(
            pred, target, torch.ones(D), average_grid=avg_grid, sum_vars=sum_v
        )
        expected_shape = (B,)
        if not avg_grid:
            expected_shape += (N,)
        if not sum_v:
            expected_shape += (D,)
        assert result.shape == expected_shape


class TestWeightedMetrics:
    """Verify weighting by pred_std."""

    def test_wmse_scaling(self):
        pred = torch.randn(2, 10, 5)
        target = torch.randn(2, 10, 5)
        std1 = torch.ones(5)
        std2 = 2 * torch.ones(5)
        r1 = wmse(pred, target, std1)
        r2 = wmse(pred, target, std2)
        assert torch.allclose(r2, r1 / 4, atol=1e-5)

    def test_wmae_scaling(self):
        pred = torch.randn(2, 10, 5)
        target = torch.randn(2, 10, 5)
        std1 = torch.ones(5)
        std2 = 2 * torch.ones(5)
        r1 = wmae(pred, target, std1)
        r2 = wmae(pred, target, std2)
        assert torch.allclose(r2, r1 / 2, atol=1e-5)


class TestCRPS:
    """Verify CRPS properties."""

    def test_crps_non_negative(self):
        pred = torch.randn(2, 10, 5)
        target = torch.randn(2, 10, 5)
        std = torch.ones(5).clamp(min=1e-6)  # ensure positive for stability
        result = crps_gauss(
            pred, target, std, sum_vars=False, average_grid=False
        )
        # CRPS is non-negative (standard definition)
        assert (result >= 0).all()

    def test_crps_zero_std_positive(self):
        pred = torch.randn(2, 10, 5)
        target = pred.clone()
        std = torch.ones(5) * 1e-6  # near-deterministic
        result = crps_gauss(pred, target, std)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()


class TestNLL:
    """Basic NLL sanity checks."""

    def test_nll_non_negative(self):
        pred = torch.randn(2, 10, 5)
        target = torch.randn(2, 10, 5)
        std = torch.ones(5).clamp(min=1e-6)
        result = nll(pred, target, std, sum_vars=False, average_grid=False)
        assert (result >= 0).all()

    def test_nll_perfect_prediction_large_std(self):
        pred = target = torch.zeros(1, 5, 2)
        std = torch.ones(2) * 10.0
        result = nll(pred, target, std)
        assert result.shape == (1,)
        assert torch.isfinite(result).all()


class TestEdgeCases:
    """Edge cases: zero/near-zero std, single grid point, broadcasting."""

    def test_single_grid_point(self):
        pred = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        target = torch.tensor([[[1.0, 2.0]]])
        result = mse(pred, target, torch.ones(2))
        assert torch.allclose(result, torch.tensor(0.0))

    def test_wmse_zero_std_produces_inf(self):
        pred = torch.ones(1, 2, 2)
        target = torch.zeros(1, 2, 2)
        std = torch.zeros(2)
        result = wmse(pred, target, std)
        assert not torch.isfinite(result).any()

    def test_wmae_zero_std_produces_inf(self):
        pred = torch.ones(1, 2, 2)
        target = torch.zeros(1, 2, 2)
        std = torch.zeros(2)
        result = wmae(pred, target, std)
        assert not torch.isfinite(result).any()
