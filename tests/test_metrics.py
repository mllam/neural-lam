# Third-party
import pytest
import torch

# First-party
from neural_lam import metrics


class TestRankHistogram:
    """Tests for rank_histogram ensemble calibration diagnostic."""

    def test_basic_shape(self):
        """Output shape is (num_bins, d_state)."""
        ensemble_pred = torch.randn(5, 4, 10, 3)
        target = torch.randn(4, 10, 3)

        bin_counts = metrics.rank_histogram(ensemble_pred, target)

        assert bin_counts.shape == (6, 3)
        assert bin_counts.dtype == torch.long

    def test_with_mask(self):
        """Masking reduces sample count."""
        ensemble_pred = torch.randn(4, 2, 10, 2)
        target = torch.randn(2, 10, 2)
        mask = torch.tensor([True] * 6 + [False] * 4)

        masked = metrics.rank_histogram(ensemble_pred, target, mask=mask)
        full = metrics.rank_histogram(ensemble_pred, target)

        assert masked.sum() == 2 * 6 * 2
        assert full.sum() == 2 * 10 * 2

    def test_conservation(self):
        """Total counts equal number of samples."""
        ensemble_pred = torch.randn(7, 3, 12, 4)
        target = torch.randn(3, 12, 4)

        bin_counts = metrics.rank_histogram(ensemble_pred, target)

        for var_idx in range(4):
            assert bin_counts[:, var_idx].sum() == 3 * 12

    def test_extreme_values(self):
        """Target outside ensemble range."""
        ensemble_pred = torch.zeros(5, 1, 1, 1)
        ensemble_pred[:, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        hist_below = metrics.rank_histogram(
            ensemble_pred, torch.tensor([[[0.0]]])
        )
        assert hist_below[0, 0] == 1

        hist_above = metrics.rank_histogram(
            ensemble_pred, torch.tensor([[[6.0]]])
        )
        assert hist_above[5, 0] == 1

    def test_calibrated_ensemble(self):
        """Same distribution produces approximately flat histogram."""
        torch.manual_seed(42)
        ensemble_pred = torch.randn(10, 10000, 1, 1)
        target = torch.randn(10000, 1, 1)

        bin_counts = metrics.rank_histogram(ensemble_pred, target)
        expected = 10000 / 11

        for bin_idx in range(11):
            assert abs(bin_counts[bin_idx, 0] - expected) < 0.15 * expected

