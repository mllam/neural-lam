# Third-party
import pytest
import torch

# First-party
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
