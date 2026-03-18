"""
Tests for spread_squared ensemble variance metric.

Verifies:
1. Mathematical correctness against known analytical values
2. Unbiased estimation property (converges to true variance)
3. Guard against single-member ensembles (S=1)
4. Output shape matches the API contract
5. Consistency with torch.var (Bessel-corrected)
6. Correct behavior with mask and reduction flags

Shape note: tests use (B, S, N, F) without an explicit T dimension.
This is intentional — the metric follows the (..., S, N, d_state) convention
where T (from Issue #335's (B, T, N, F) shape) is just another batch dim
folded into (...). In ar_model.py, B and T are already flattened before
metrics are called, so (B, S, N, F) accurately reflects real call shapes.
"""

# Third-party
import pytest
import torch

# First-party
from neural_lam.metrics import spread_squared


class TestSpreadSquared:
    """Tests for the spread_squared (ensemble variance) metric."""

    def test_known_value_two_members(self):
        """
        For pred = [1.0, 3.0], unbiased variance = 2.0.
        mean = 2.0, deviations = [-1, 1], sum of sq = 2, / (2-1) = 2.0
        """
        # Shape: (B=1, S=2, N=1, d_state=1)
        # T is folded into B per the (...) convention
        pred = torch.tensor([[[[1.0]], [[3.0]]]])
        target = torch.zeros(1, 1, 1)  # (B, N, d_state) — unused
        pred_std = torch.ones(1, 1, 1)  # unused

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )

        # With average_grid=False, sum_vars=False → shape (B, N, d_state)
        assert result.shape == (
            1,
            1,
            1,
        ), f"Expected shape (1,1,1), got {result.shape}"
        assert torch.allclose(
            result, torch.tensor([[[2.0]]])
        ), f"Expected 2.0, got {result.item()}"

    def test_known_value_three_members(self):
        """
        For pred = [2.0, 4.0, 6.0], mean=4.0,
        unbiased var = ((2-4)^2 + (4-4)^2 + (6-4)^2) / (3-1) = 4.0.
        """
        # Shape: (B=1, S=3, N=1, d_state=1)
        # T is folded into B per the (...) convention
        pred = torch.tensor([[[[2.0]], [[4.0]], [[6.0]]]])
        target = torch.zeros(1, 1, 1)
        pred_std = torch.ones(1, 1, 1)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )

        assert torch.allclose(
            result, torch.tensor([[[4.0]]])
        ), f"Expected 4.0, got {result.item()}"

    def test_unbiased_estimation_large_ensemble(self):
        """
        For a large ensemble drawn from N(0,1), the unbiased sample
        variance should converge to the true variance (1.0).
        """
        torch.manual_seed(42)
        S = 10000
        # Shape: (B=1, S=10000, N=1, d_state=1)
        # T is folded into B per the (...) convention
        pred = torch.randn(1, S, 1, 1)
        target = torch.zeros(1, 1, 1)
        pred_std = torch.ones(1, 1, 1)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )

        assert (
            torch.abs(result.squeeze() - 1.0) < 0.05
        ), f"Expected ~1.0 for N(0,1) variance, got {result.item()}"

    def test_rejects_single_member(self):
        """S=1 must raise AssertionError — single-member variance
        is undefined."""
        pred = torch.randn(1, 1, 1, 1)  # S=1
        target = torch.zeros(1, 1, 1)
        pred_std = torch.ones(1, 1, 1)

        with pytest.raises(AssertionError, match="more than 1 member"):
            spread_squared(
                pred,
                target,
                pred_std,
                mask=None,
                average_grid=False,
                sum_vars=False,
            )

    def test_output_shape_no_reduction(self):
        """
        With average_grid=False, sum_vars=False, output shape should
        be (..., N, d_state) with ensemble dim reduced.

        T is a batch dimension folded into (...); pred is (B*T, S, N, F).
        """
        B, S, N, F = 4, 8, 100, 3
        pred = torch.randn(B, S, N, F)  # (..., S, N, d_state)
        target = torch.randn(B, N, F)
        pred_std = torch.ones(F)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )

        # S reduced, rest preserved: (B, N, F)
        assert result.shape == (
            B,
            N,
            F,
        ), f"Expected ({B},{N},{F}), got {result.shape}"

    def test_output_shape_with_reduction(self):
        """
        With average_grid=True, sum_vars=True (defaults), output
        should reduce N and d_state dims.

        T is a batch dimension folded into (...); pred is (B*T, S, N, F).
        """
        B, S, N, F = 2, 5, 50, 4
        pred = torch.randn(B, S, N, F)
        target = torch.randn(B, N, F)
        pred_std = torch.ones(F)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=True,
            sum_vars=True,
        )

        # S reduced, N averaged, F summed: (B,)
        assert result.shape == (B,), f"Expected ({B},), got {result.shape}"

    def test_matches_torch_var(self):
        """
        Verify that spread_squared (without grid/var reduction) matches
        torch.var with Bessel's correction across the ensemble dimension.
        """
        torch.manual_seed(123)
        B, S, N, F = 2, 5, 10, 4
        # T folded into B per (...) convention
        pred = torch.randn(B, S, N, F)
        target = torch.randn(B, N, F)
        pred_std = torch.ones(F)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )
        # ens_dim=-3 means dim=1 for 4D input (B, S, N, F)
        expected = torch.var(pred, dim=-3, unbiased=True)

        assert torch.allclose(
            result, expected, atol=1e-6
        ), "spread_squared does not match torch.var(unbiased=True)"

    def test_zero_spread(self):
        """
        If all ensemble members are identical, variance should be 0.
        """
        # Shape: (B=1, S=3, N=1, d_state=1) — all members = 5.0
        pred = torch.full((1, 3, 1, 1), 5.0)
        target = torch.zeros(1, 1, 1)
        pred_std = torch.ones(1, 1, 1)

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=None,
            average_grid=False,
            sum_vars=False,
        )

        assert torch.allclose(
            result, torch.tensor([[[0.0]]])
        ), f"Expected 0.0 for identical members, got {result.item()}"

    def test_with_mask(self):
        """
        Verify that the boolean mask correctly filters grid nodes.
        """
        # Shape: (B=1, S=3, N=4, d_state=1)
        pred = torch.randn(1, 3, 4, 1)
        target = torch.randn(1, 4, 1)
        pred_std = torch.ones(1)
        mask = torch.tensor([True, False, True, False])  # keep 2 of 4 nodes

        result = spread_squared(
            pred,
            target,
            pred_std,
            mask=mask,
            average_grid=False,
            sum_vars=False,
        )

        # N filtered from 4 to 2
        assert result.shape == (
            1,
            2,
            1,
        ), f"Expected shape (1,2,1) with mask, got {result.shape}"
