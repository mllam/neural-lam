"""Tests for evaluation metrics in neural_lam.metrics."""

# Third-party
import numpy as np
import pytest
import torch

# First-party
from neural_lam import metrics


def test_metric_retrieval():
    """Verify metric lookup by name."""
    assert metrics.get_metric("mse") == metrics.mse
    assert metrics.get_metric("mbe") == metrics.mbe
    assert metrics.get_metric("MAE") == metrics.mae

    with pytest.raises(AssertionError, match="Unknown metric"):
        metrics.get_metric("nonexistent_metric")


def test_mean_bias_error():
    """Verify MBE computes signed difference correctly."""
    pred = torch.tensor([[[2.0, 4.0], [3.0, 5.0]]])  # (1, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])  # (1, 2, 2)
    dummy_std = torch.ones_like(pred)

    bias = metrics.mbe(
        pred, target, dummy_std, average_grid=True, sum_vars=True
    )
    # (2-1) + (4-2) = 1+2 = 3 for node 0
    # (3-1) + (5-2) = 2+3 = 5 for node 1
    # Average over 2 nodes = (3+5)/2 = 4.0
    assert torch.isclose(bias, torch.tensor(4.0))


def test_radial_power_spectrum_2d_shape_and_values():
    """Verify 2D radial FFT power spectrum calculation."""
    # Constant field -> all energy at k=0 (which is excluded from non-zero bins)
    constant_field = torch.ones(1, 32, 32)
    psd_const, k_bins = metrics.radial_power_spectrum_2d(constant_field)

    assert len(k_bins) == 16
    assert psd_const.shape == (1, 16)
    assert torch.allclose(psd_const, torch.zeros_like(psd_const), atol=1e-5)

    # Sine wave field along x -> energy concentrated at specific wavenumber
    y = torch.linspace(0, 2 * torch.pi, 32)
    x = torch.linspace(0, 2 * torch.pi, 32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    # Wave with wavenumber 4
    sine_field = torch.sin(4 * grid_x).unsqueeze(0)

    psd_sine, k_bins = metrics.radial_power_spectrum_2d(sine_field)
    assert psd_sine.shape == (1, 16)
    # Peak should be at k=4 (index 3 since k_bins starts at 1)
    peak_k = k_bins[torch.argmax(psd_sine[0])]
    assert peak_k.item() == 4


def test_dct_2d_transform_and_power_spectrum():
    """Verify 2D DCT-II transform and radially averaged DCT power spectrum."""
    field = torch.randn(2, 16, 16)
    dct = metrics.DiscreteCosineTransform2D(16, 16)
    coeffs = dct(field)
    assert coeffs.shape == (2, 16, 16)

    k_centers, psd = metrics.dct_power_spectrum_2d(field)
    assert len(k_centers) == 8
    assert psd.shape == (2, 8)
    assert (psd >= 0.0).all()


def test_fractions_skill_score_2d():
    """Verify scale-dependent Fractions Skill Score (FSS)."""
    # Identical fields -> FSS should be exactly 1.0
    field = torch.tensor([[5.0, 15.0], [15.0, 5.0]])
    fss_perfect = metrics.fractions_skill_score_2d(
        field, field, threshold=10.0, kernel_size=1
    )
    assert torch.isclose(fss_perfect, torch.tensor(1.0))

    # Completely disjoint fields -> FSS is 0.0 at kernel_size=1
    target = torch.tensor([[15.0, 5.0], [5.0, 15.0]])
    fss_disjoint = metrics.fractions_skill_score_2d(
        field, target, threshold=10.0, kernel_size=1
    )
    assert torch.isclose(fss_disjoint, torch.tensor(0.0), atol=1e-5)


def test_spectral_collapse_and_hallucination_index():
    """Verify SCR and Hallucination Index."""
    psd_model = torch.tensor([10.0, 5.0, 1.0])
    psd_targ = torch.tensor([10.0, 5.0, 2.0])

    scr = metrics.spectral_collapse_ratio(psd_model, psd_targ)
    assert torch.isclose(scr[0], torch.tensor(1.0))
    assert torch.isclose(scr[2], torch.tensor(0.5))

    hi = metrics.hallucination_index(scr_fine=0.9, fss_fine=0.2)
    assert np.isclose(hi, 0.9 * (1.0 - 0.2))
