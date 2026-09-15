# Third-party
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    """Flux stats distributed gather: stack per-rank batch scalars into 1-D."""

    def _make_gathered(self, world_size, n_batches):
        return [
            [torch.tensor(float(r * 10 + b)) for b in range(n_batches)]
            for r in range(world_size)
        ]

    def test_fix_shape(self):
        """Stack flattens per-rank scalars into a 1-D tensor."""
        ws, nb = 4, 3
        gathered = self._make_gathered(ws, nb)
        result = torch.cat([torch.stack(rf) for rf in gathered])
        assert result.shape == (ws * nb,)

    def test_fix_mean(self):
        """Global mean is the mean over all per-rank batch scalars."""
        gathered = [
            [torch.tensor(0.0), torch.tensor(2.0)],
            [torch.tensor(4.0), torch.tensor(6.0)],
        ]
        result = torch.cat([torch.stack(rf) for rf in gathered])
        assert torch.isclose(torch.mean(result), torch.tensor(3.0))


# -- Bug 2: diff stats wrong shape -----------------------------------------


class TestDiffStatsShape:
    """Diff stats distributed gather: contiguous slice preserves (N, d_f)."""

    def test_fix_shape(self):
        """Slicing the gathered tensor preserves the feature dimension."""
        d_f, total, n_orig = 17, 100, 80
        data = torch.randn(total, d_f)

        result = data[:n_orig]
        assert result.shape == (n_orig, d_f)

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


# -- Bug 3: positional depadding after distributed gather ------------------


class _IndexEchoDataset:
    """Dataset stub whose items echo their own index."""

    def __init__(self, n_samples: int):
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx)


class TestRealSampleMasks:
    """Per-rank masks drop exactly the padded rows the DataLoader yields."""

    @pytest.mark.parametrize(
        "total_samples, world_size, batch_size",
        [
            (16, 4, 4),  # no padding
            (103, 4, 8),  # one padded row, on rank 3 only
            (101, 4, 8),  # old prefix slice kept pads 101, 102, dropped 95, 99
            (97, 4, 8),  # each pad is alone in a 1-row last batch (empty mask)
            (10, 4, 4),  # per-rank count below batch_size
        ],
    )
    def test_masked_batches_recover_each_real_sample_once(
        self, total_samples, world_size, batch_size
    ):
        """Pads echo total_samples - 1, so a leaked pad shows as a duplicate."""
        ds = PaddedWeatherDataset(
            _IndexEchoDataset(total_samples), world_size, batch_size
        )
        recovered = []
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=False
            )
            loader = DataLoader(ds, batch_size, sampler=sampler)
            masks = ds.real_sample_masks(sampler)
            assert len(masks) == len(loader)
            for batch, mask in zip(loader, masks):
                recovered.extend(batch[mask].tolist())

        assert sorted(recovered) == list(range(total_samples))
