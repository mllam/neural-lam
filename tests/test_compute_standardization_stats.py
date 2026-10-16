# Standard library
from datetime import timedelta

# Third-party
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# First-party
from neural_lam.datastore.npyfilesmeps import compute_standardization_stats
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


# -- Flux stats aggregation ------------------------------------------------


class _FluxIsIndexDataset:
    """Dataset whose sample ``i`` has state and flux equal to ``i``."""

    def __init__(self, **_):
        self._n = 101

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        state = torch.full((65, 1, 1), float(idx))
        forcing = torch.full((2, 3, 1), float(idx))
        return state[:2], state[2:], forcing, torch.zeros(1)


class TestFluxStats:
    """Flux statistics weight samples rather than DataLoader batches."""

    def test_main_weights_short_final_batch_by_sample(
        self, tmp_path, monkeypatch
    ):
        """The generated flux stats match all 101 synthetic samples."""
        monkeypatch.setattr(
            compute_standardization_stats,
            "init_datastore",
            lambda **_: object(),
        )
        monkeypatch.setattr(
            compute_standardization_stats,
            "WeatherDataset",
            _FluxIsIndexDataset,
        )

        compute_standardization_stats.main(
            datastore_config_path=tmp_path / "config.yaml",
            batch_size=8,
            step_length=timedelta(hours=1),
            n_workers=0,
            distributed=False,
        )

        samples = torch.arange(101).float()
        expected = torch.stack((samples.mean(), samples.std(unbiased=False)))
        flux_stats = torch.load(
            tmp_path / "static" / "flux_stats.pt", weights_only=True
        )
        torch.testing.assert_close(flux_stats, expected)

    def test_main_excludes_distributed_padding(self, tmp_path, monkeypatch):
        """A padded row is removed before distributed flux gathering."""
        monkeypatch.setattr(
            compute_standardization_stats,
            "init_datastore",
            lambda **_: object(),
        )
        monkeypatch.setattr(
            compute_standardization_stats,
            "WeatherDataset",
            _FluxIsIndexDataset,
        )
        monkeypatch.setattr(
            compute_standardization_stats, "get_rank", lambda: 0
        )
        monkeypatch.setattr(
            compute_standardization_stats, "get_world_size", lambda: 4
        )
        monkeypatch.setattr(
            compute_standardization_stats,
            "setup",
            lambda rank, world_size: None,
        )
        monkeypatch.setattr(
            compute_standardization_stats.torch.cuda,
            "is_available",
            lambda: False,
        )

        def rank_with_padding(dataset, **_):
            return DistributedSampler(
                dataset, num_replicas=4, rank=3, shuffle=False
            )

        def gather_local_rank(gathered, value):
            gathered[0] = value
            empty_shape = (0, *value.shape[1:])
            for rank in range(1, len(gathered)):
                gathered[rank] = value.new_empty(empty_shape)

        monkeypatch.setattr(
            compute_standardization_stats,
            "DistributedSampler",
            rank_with_padding,
        )
        monkeypatch.setattr(
            compute_standardization_stats.dist,
            "all_gather_object",
            gather_local_rank,
        )
        monkeypatch.setattr(
            compute_standardization_stats.dist, "barrier", lambda: None
        )
        monkeypatch.setattr(
            compute_standardization_stats.dist,
            "destroy_process_group",
            lambda: None,
        )

        compute_standardization_stats.main(
            datastore_config_path=tmp_path / "config.yaml",
            batch_size=8,
            step_length=timedelta(hours=1),
            n_workers=0,
            distributed=True,
        )

        samples = torch.arange(3, 101, 4).float()
        expected = torch.stack((samples.mean(), samples.std(unbiased=False)))
        flux_stats = torch.load(
            tmp_path / "static" / "flux_stats.pt", weights_only=True
        )
        torch.testing.assert_close(flux_stats, expected)


# -- Bug 2: positional depadding after distributed gather ------------------


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
