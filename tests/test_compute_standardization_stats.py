# Third-party
import pytest
import torch

# First-party
from neural_lam.datastore.npyfilesmeps.compute_standardization_stats import (
    PaddedWeatherDataset,
)


class _StubDataset:
    """Minimal dataset stub."""

    def __init__(self, n_samples: int):
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return torch.zeros(4)


# -- PaddedWeatherDataset ---------------------------------------------------


class TestPaddedWeatherDataset:
    """Tests for PaddedWeatherDataset helper."""

    def test_original_indices(self):
        """get_original_indices must return [0, ..., N-1]."""
        ds = PaddedWeatherDataset(_StubDataset(10), world_size=4, batch_size=4)
        assert list(ds.get_original_indices()) == list(range(10))

    def test_padded_length(self):
        """Pad total to next multiple of world_size."""
        ds = PaddedWeatherDataset(_StubDataset(10), world_size=4, batch_size=4)
        assert len(ds) == 12

    def test_no_padding_needed(self):
        """No padding when evenly divisible."""
        ds = PaddedWeatherDataset(_StubDataset(16), world_size=4, batch_size=4)
        assert len(ds) == 16

    def test_padded_item_returns_last_real(self):
        """Padded indices return the last real sample."""
        ds = PaddedWeatherDataset(_StubDataset(10), world_size=4, batch_size=4)
        item_real = ds[9]
        for padded_idx in range(10, len(ds)):
            assert torch.equal(item_real, ds[padded_idx])


# -- Bug 1: flux stats IndexError ------------------------------------------


class TestFluxStatsGather:
    """Flux stats distributed gather fix."""

    def _make_gathered(self, world_size, n_batches):
        return [
            [torch.tensor(float(r * 10 + b)) for b in range(n_batches)]
            for r in range(world_size)
        ]

    def test_old_code_raises_indexerror(self):
        """torch.tensor(gathered)[i] crashes for i >= ws."""
        ws, nb, n_orig = 2, 3, 50
        gathered = self._make_gathered(ws, nb)
        old = torch.tensor(gathered)
        assert old.shape == (ws, nb)
        with pytest.raises(IndexError):
            _ = [old[i] for i in range(n_orig)]

    def test_fix_shape(self):
        """Flatten only real per-rank flux scalars into 1-D tensor."""
        ws, nb = 4, 3
        n_real_per_rank = 2
        gathered = self._make_gathered(ws, nb)
        result = torch.cat(
            [torch.stack(rf[:n_real_per_rank]) for rf in gathered]
        )
        assert result.shape == (ws * n_real_per_rank,)

    def test_fix_mean(self):
        """Padded flux scalars do not contribute to the global mean."""
        gathered = [
            [torch.tensor(0.0), torch.tensor(100.0)],
            [torch.tensor(2.0), torch.tensor(200.0)],
        ]
        n_real_per_rank = 1
        result = torch.cat(
            [torch.stack(rf[:n_real_per_rank]) for rf in gathered]
        )
        assert torch.isclose(torch.mean(result), torch.tensor(1.0))

    def test_fix_single_rank(self):
        """Single-rank edge case."""
        gathered = [[torch.tensor(1.0), torch.tensor(2.0)]]
        n_real_per_rank = 2
        result = torch.cat(
            [torch.stack(rf[:n_real_per_rank]) for rf in gathered]
        )
        assert result.shape == (2,)

    def test_fix_large_world_size(self):
        """8 GPUs."""
        ws, nb = 8, 5
        n_real_per_rank = 4
        gathered = self._make_gathered(ws, nb)
        result = torch.cat(
            [torch.stack(rf[:n_real_per_rank]) for rf in gathered]
        )
        assert result.shape == (ws * n_real_per_rank,)


# -- Bug 2: diff stats wrong shape -----------------------------------------


class TestDiffStatsShape:
    """Diff stats distributed shape fix."""

    def test_old_code_wrong_shape(self):
        """List-comp gives flat 1-D via torch.cat."""
        d_f, total, n_orig = 17, 100, 80
        data = torch.randn(total, d_f)
        indices = list(range(n_orig))

        old = [data[i] for i in indices]
        assert old[0].shape == (d_f,)
        flat = torch.cat(old, dim=0)
        # (n_orig * d_f,) — wrong
        assert flat.shape == (n_orig * d_f,)
        # scalar mean — wrong
        assert torch.mean(flat, dim=0).shape == ()

    def test_fix_shape(self):
        """Contiguous slice preserves (N, d_f)."""
        d_f, total, n_orig = 17, 100, 80
        data = torch.randn(total, d_f)

        result = data[:n_orig]
        assert result.shape == (n_orig, d_f)

    def test_fix_mean_shape(self):
        """save_stats produces (d_f,) mean."""
        d_f, total, n_orig = 17, 100, 80
        data = torch.randn(total, d_f)

        result = data[:n_orig]
        assert torch.mean(result, dim=0).shape == (d_f,)

    def test_fix_preserves_values(self):
        """Contiguous slice selects the expected rows."""
        d_f, total = 5, 10
        data = torch.arange(total * d_f, dtype=torch.float32).view(total, d_f)

        result = data[:4]
        expected = torch.stack([data[0], data[1], data[2], data[3]])
        assert torch.equal(result, expected)

    def test_slice_handles_larger_step_length(self):
        """Step lengths above one select distinct diff rows."""
        d_f = 3
        n_samples = 4
        step_int = 3
        data = torch.arange(20 * d_f, dtype=torch.float32).view(20, d_f)
        old_indices = [i // step_int for i in range(n_samples * step_int)]

        result = data[: n_samples * step_int]
        old_result = data[old_indices]

        assert torch.equal(result, data[:12])
        assert not torch.equal(result, old_result)

    def test_fix_all_indices(self):
        """No padding — identity selection."""
        d_f, n = 8, 20
        data = torch.randn(n, d_f)

        result = data[:n]
        assert result.shape == (n, d_f)
        assert torch.equal(result, data)
