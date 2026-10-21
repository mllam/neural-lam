"""Tests for evaluation metrics in neural_lam.metrics."""

# Third-party
import pytest
import torch

# First-party
from neural_lam import metrics


def test_metric_retrieval():
    """Verify metric lookup by name and excluded loss metrics."""
    assert metrics.get_metric("mse") == metrics.mse
    assert metrics.get_metric("MAE") == metrics.mae
    assert metrics.get_metric("wmse") == metrics.wmse

    # mbe is an evaluation-only metric, not an allowed training loss
    with pytest.raises(AssertionError, match="Unknown metric"):
        metrics.get_metric("mbe")

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


def test_dct_2d_transform_and_power_spectrum():
    """Verify 2D DCT-II transform and radially averaged DCT power spectrum."""
    field = torch.randn(2, 16, 16)
    dct = metrics.DiscreteCosineTransform2D(16, 16)
    coeffs = dct(field)
    assert coeffs.shape == (2, 16, 16)

    energy_field = (field**2).sum(dim=(-2, -1))
    energy_coeffs = (coeffs**2).sum(dim=(-2, -1))
    assert torch.allclose(energy_field, energy_coeffs, atol=1e-4)

    k_centers, psd = metrics.dct_power_spectrum_2d(field)
    assert len(k_centers) == 8
    assert psd.shape == (2, 8)
    assert (psd >= 0.0).all()


def test_log_spectral_distance():
    """Verify Log-Spectral Distance calculation."""
    # Identical spectra -> LSD = 0.0 dB
    psd = torch.tensor([1.0, 2.0, 0.5, 0.1])
    lsd_identical = metrics.log_spectral_distance(psd, psd)
    assert torch.isclose(lsd_identical, torch.tensor(0.0), atol=1e-5)

    # 10x ratio -> 10 * log10(10) = 10 dB
    psd_10x = psd * 10.0
    lsd_10x = metrics.log_spectral_distance(psd, psd_10x)
    assert torch.isclose(lsd_10x, torch.tensor(10.0), atol=1e-4)


def test_fractions_skill_score_2d():
    """Verify scale-dependent Fractions Skill Score (FSS)."""
    field = torch.tensor([[5.0, 15.0], [15.0, 5.0]])
    fss_perfect = metrics.fractions_skill_score_2d(
        field, field, threshold=10.0, kernel_size=1
    )
    assert torch.isclose(fss_perfect, torch.tensor(1.0))

    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    fss_masked = metrics.fractions_skill_score_2d(
        field, field, threshold=10.0, kernel_size=1, mask=mask
    )
    assert torch.isclose(fss_masked, torch.tensor(1.0))

    target = torch.tensor([[15.0, 5.0], [5.0, 15.0]])
    fss_disjoint = metrics.fractions_skill_score_2d(
        field, target, threshold=10.0, kernel_size=1
    )
    assert torch.isclose(fss_disjoint, torch.tensor(0.0), atol=1e-5)

    # Enforce odd kernel sizes
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        metrics.fractions_skill_score_2d(
            field, target, threshold=10.0, kernel_size=2
        )

    # Over-sized kernel exceeding interior mask -> NaN
    tiny_mask = torch.zeros(10, 10)
    tiny_mask[4:6, 4:6] = 1.0
    fss_nan = metrics.fractions_skill_score_2d(
        torch.zeros(10, 10),
        torch.ones(10, 10),
        threshold=0.5,
        kernel_size=5,
        mask=tiny_mask,
    )
    assert torch.isnan(fss_nan)


def test_contingency_and_skill_scores():
    """Verify contingency table, Frequency Bias, and Equitable Threat Score."""
    pred = torch.tensor([12.0, 8.0, 14.0, 2.0])
    target = torch.tensor([11.0, 9.0, 4.0, 1.0])
    thresh = 10.0

    # Events (>= 10):
    # pred:   [True, False, True, False]
    # target: [True, False, False, False]
    # idx 0: pred=T, target=T -> Hit
    # idx 1: pred=F, target=F -> Correct Negative
    # idx 2: pred=T, target=F -> False Alarm
    # idx 3: pred=F, target=F -> Correct Negative
    h, f, m, cn = metrics.contingency_table(pred, target, thresh)
    assert h.item() == 1.0
    assert f.item() == 1.0
    assert m.item() == 0.0
    assert cn.item() == 2.0

    fb = metrics.frequency_bias(pred, target, thresh)
    # (1 + 1) / (1 + 0) = 2.0
    assert torch.isclose(fb, torch.tensor(2.0))

    # Perfect forecast
    fb_perf = metrics.frequency_bias(pred, pred, thresh)
    ets_perf = metrics.equitable_threat_score(pred, pred, thresh)
    assert torch.isclose(fb_perf, torch.tensor(1.0))
    assert torch.isclose(ets_perf, torch.tensor(1.0))
