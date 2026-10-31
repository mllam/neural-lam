# Standard library
from datetime import timedelta

# Third-party
import pytest
import torch
import torch.multiprocessing as mp
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

    n_samples = 101

    def __init__(self, **_):
        self._n = self.n_samples

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        state = torch.full((65, 1, 1), float(idx))
        forcing = torch.full((2, 3, 1), float(idx))
        return state[:2], state[2:], forcing, torch.zeros(0), torch.zeros(1)


def _run_distributed_stats(rank, world_size, n_samples, config_path, init_path):
    """Run one real Gloo rank of the standardization-statistics job."""
    compute_standardization_stats.init_datastore = lambda **_: object()
    _FluxIsIndexDataset.n_samples = n_samples
    compute_standardization_stats.WeatherDataset = _FluxIsIndexDataset
    compute_standardization_stats.get_rank = lambda: rank
    compute_standardization_stats.get_world_size = lambda: world_size
    compute_standardization_stats.torch.cuda.is_available = lambda: False

    def setup(_rank, _world_size):
        compute_standardization_stats.dist.init_process_group(
            "gloo",
            init_method=init_path.as_uri(),
            rank=_rank,
            world_size=_world_size,
            timeout=timedelta(seconds=60),
        )

    compute_standardization_stats.setup = setup
    compute_standardization_stats.main(
        datastore_config_path=config_path,
        batch_size=8,
        step_length=timedelta(hours=1),
        n_workers=0,
        distributed=True,
    )


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

    @pytest.mark.parametrize(
        "n_samples, world_size",
        [(97, 4)],
    )
    def test_main_gathers_short_batches_without_distributed_padding(
        self, tmp_path, n_samples, world_size
    ):
        """Gather each sample once despite short batches and padding."""
        mp.spawn(
            _run_distributed_stats,
            args=(
                world_size,
                n_samples,
                tmp_path / "config.yaml",
                tmp_path / "dist_init",
            ),
            nprocs=world_size,
            join=True,
        )

        samples = torch.arange(n_samples).float()
        expected = torch.stack((samples.mean(), samples.std(unbiased=False)))
        flux_stats = torch.load(
            tmp_path / "static" / "flux_stats.pt", weights_only=True
        )
        torch.testing.assert_close(flux_stats, expected)
        parameter_mean = torch.load(
            tmp_path / "static" / "parameter_mean.pt", weights_only=True
        )
        parameter_std = torch.load(
            tmp_path / "static" / "parameter_std.pt", weights_only=True
        )
        torch.testing.assert_close(parameter_mean, expected[0].unsqueeze(0))
        torch.testing.assert_close(parameter_std, expected[1].unsqueeze(0))
        diff_mean = torch.load(
            tmp_path / "static" / "diff_mean.pt", weights_only=True
        )
        diff_std = torch.load(
            tmp_path / "static" / "diff_std.pt", weights_only=True
        )
        torch.testing.assert_close(diff_mean, torch.zeros(1))
        torch.testing.assert_close(diff_std, torch.zeros(1))


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
