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
    sample_moments,
    save_stats,
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


# -- std from moments: large mean relative to std ---------------------------


def _saved_std(tmp_path, x):
    """Run per-sample moments of ``x`` through save_stats, return the std."""
    means, squares = sample_moments(x)
    save_stats(tmp_path, [means], [squares], [], [], "parameter")
    return torch.load(tmp_path / "parameter_std.pt", weights_only=True)


class TestStdFromMoments:
    """Std must stay accurate when the mean is large compared to the std."""

    @pytest.mark.parametrize(
        "mean,std",
        [(280.0, 10.0), (1e5, 1e3), (1e5, 100.0), (1e5, 20.0), (1e5, 5.0)],
    )
    def test_matches_float64_reference(self, tmp_path, mean, std):
        """E[x^2] - E[x]^2 in float32 gives 9-60% errors or NaN here."""
        g = torch.Generator().manual_seed(0)
        x = (mean + std * torch.randn(8, 5, 200, 2, generator=g)).float()
        expected = x.double().std(dim=(0, 1, 2), unbiased=False)
        result = _saved_std(tmp_path, x)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result.double(), expected, rtol=1e-3, atol=0.0
        )

    @pytest.mark.parametrize("value", [1e5, 100000.1])
    def test_constant_field_has_negligible_std(self, tmp_path, value):
        """A constant field must not give NaN or a std of the float32 noise
        level (tens, in float32), only below one float32 step of the data."""
        x = torch.full((4, 5, 50, 1), value)
        std = _saved_std(tmp_path, x).item()
        assert std <= torch.finfo(torch.float32).eps * value

    def test_saved_as_float32(self, tmp_path):
        """Stats are applied to float32 batches, float64 would upcast them."""
        x = torch.randn(4, 5, 50, 3) + 100.0
        means, squares = sample_moments(x)
        save_stats(
            tmp_path, [means], [squares], [means[:, 0]], [squares[:, 0]], "p"
        )
        assert torch.load(tmp_path / "p_mean.pt").dtype == torch.float32
        assert torch.load(tmp_path / "p_std.pt").dtype == torch.float32
        assert torch.load(tmp_path / "flux_stats.pt").dtype == torch.float32

    def test_flux_stats(self, tmp_path):
        """Flux mean and std come out of the same float64 path."""
        g = torch.Generator().manual_seed(0)
        flux = (1e5 + 20.0 * torch.randn(4 * 5 * 50, generator=g)).float()
        per_sample = flux.view(4, -1).double()
        save_stats(
            tmp_path,
            [torch.zeros(4, 1)],
            [torch.zeros(4, 1)],
            [per_sample.mean(dim=1)],
            [(per_sample**2).mean(dim=1)],
            "p",
        )
        flux_mean, flux_std = torch.load(tmp_path / "flux_stats.pt")
        torch.testing.assert_close(
            flux_std.double(),
            flux.double().std(unbiased=False),
            rtol=1e-3,
            atol=0,
        )
        torch.testing.assert_close(
            flux_mean.double(), flux.double().mean(), rtol=1e-6, atol=0
        )

    def test_inconsistent_moments_raise(self, tmp_path):
        """E[x^2] < E[x]^2 is impossible, so it must not be written out."""
        means = torch.tensor([[3.0]], dtype=torch.float64)
        squares = torch.tensor([[1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="Negative variance"):
            save_stats(tmp_path, [means], [squares], [], [], "parameter")
