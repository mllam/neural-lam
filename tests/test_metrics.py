# Third-party
import torch

# First-party
from neural_lam.metrics import crps_ens, spread_squared


def test_spread_squared():
    # Simulate ensemble prediction: (batch, ensemble_members, nodes, features)
    # Shape: (1, 4, 1, 1). Values: 2, 4, 4, 6
    pred = torch.tensor([[[[2.0]], [[4.0]], [[4.0]], [[6.0]]]])
    target = torch.tensor([[[4.0]]])  # Dummy target

    # Variance of [2, 4, 4, 6] with Bessel's correction (ddof=1) is 8/3
    # torch.var by default uses unbiased estimator
    result = spread_squared(
        pred, target, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )

    expected_variance = torch.tensor([[[2.0, 4.0, 4.0, 6.0]]]).var()

    assert torch.isclose(result, expected_variance, atol=1e-5), (
        f"Expected {expected_variance}, got {result}"
    )


def test_crps_ens_small_ensemble():
    # Test crps_ens on a small ensemble (num_ens < 10)
    # Shape: (1, 3, 1, 1) - 3 ensemble members
    pred = torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[3.0]]])  # target is 3

    # Calculate expected CRPS manually (unbiased estimator, default)
    # X = {1, 2, 5}, y = 3, M = 3
    # E|X - y| = (2 + 1 + 2) / 3 = 5/3
    # Rank-based: diff_factor = 1/(M-1) = 1/2
    # Unbiased CRPS ≈ 1/3
    result = crps_ens(
        pred, target, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )

    expected_crps = torch.tensor(1.0 / 3.0)

    assert torch.isclose(result, expected_crps, atol=1e-5), (
        f"Expected {expected_crps}, got {result}"
    )


def test_crps_ens_large_ensemble():
    # Test crps_ens on a large ensemble (num_ens >= 10)
    pred = torch.arange(10.0).view(1, 10, 1, 1)  # Members 0 to 9
    target = torch.tensor([[[4.5]]])  # Target in the middle

    result = crps_ens(
        pred, target, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )

    # For {0..9}, target 4.5:
    # MAE = 2.5, pairwise sum (i!=j) = 330
    # Unbiased: pair_term = 330 / (2*10*9) = 330/180
    # CRPS = 2.5 - 330/180
    expected_crps = torch.tensor(2.5 - (330 / 180))

    assert torch.isclose(result, expected_crps, atol=1e-5), (
        f"Expected {expected_crps}, got {result}"
    )


def test_crps_ens_single_member():
    """N=1 reduces to MAE — verify sum_vars and pred_std are correctly
    passed through. Previously both were silently dropped causing a
    TypeError crash and incorrect output shapes."""
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])  # shape (1, 1, 1, 3)
    target = torch.tensor([[[2.0, 2.0, 2.0]]])  # shape (1, 1, 3)
    pred_std = torch.ones_like(pred)

    result_summed = crps_ens(
        pred, target, pred_std, sum_vars=True, ens_dim=1
    )
    result_not_summed = crps_ens(
        pred, target, pred_std, sum_vars=False, ens_dim=1
    )

    assert result_not_summed.shape[-1] == 3, (
        "sum_vars=False should preserve feature dimension in N=1 path"
    )
    assert result_summed.shape != result_not_summed.shape, (
        "sum_vars flag ignored in N=1 path"
    )


def test_crps_ens_single_member_none_pred_std():
    """N=1 with pred_std=None should not crash."""
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])  # shape (1, 1, 1, 3)
    target = torch.tensor([[[2.0, 2.0, 2.0]]])  # shape (1, 1, 3)

    # This previously crashed with: TypeError: ones_like(): argument
    # must be Tensor, not NoneType
    result = crps_ens(
        pred, target, pred_std=None, sum_vars=True, ens_dim=1
    )
    assert result.numel() == 1, "Should return a scalar"


def test_crps_ens_single_member_mask_consistency():
    """Mask is correctly applied in N=1 path (from Panchadip's review)."""
    pred = torch.tensor([[[[1.0, 2.0, 3.0]]]])  # (1, 1, 1, 3)
    target = torch.tensor([[[2.0, 2.0, 2.0]]])  # (1, 1, 3)
    pred_std = torch.ones_like(pred)
    mask = torch.tensor([True])  # single grid node, keep it

    result = crps_ens(
        pred, target, pred_std, mask=mask, sum_vars=False, ens_dim=1
    )
    # Should produce a result with shape (..., 3) since sum_vars=False
    assert result.shape[-1] == 3, "masked result should preserve features"


def test_crps_ens_biased():
    """Verify biased CRPS estimator produces a distinct value from
    the unbiased one."""
    pred = torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[3.0]]])

    result_biased = crps_ens(
        pred, target, estimator="biased", ens_dim=1,
        average_grid=True, sum_vars=True,
    )
    result_unbiased = crps_ens(
        pred, target, estimator="unbiased", ens_dim=1,
        average_grid=True, sum_vars=True,
    )

    # Biased uses diff_factor = 1/M = 1/3
    # Unbiased uses diff_factor = 1/(M-1) = 1/2
    # They should differ
    assert not torch.isclose(result_biased, result_unbiased, atol=1e-5), (
        "Biased and unbiased CRPS should differ"
    )
    # Biased CRPS >= unbiased CRPS (biased penalises spread less)
    assert result_biased >= result_unbiased, (
        "Biased CRPS should be >= unbiased CRPS"
    )


def test_crps_ens_almost_fair():
    """Verify almost-fair CRPS with alpha=0 matches unbiased."""
    pred = torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[3.0]]])

    result_af = crps_ens(
        pred, target, estimator="almost-fair", afc_alpha=0.0,
        ens_dim=1, average_grid=True, sum_vars=True,
    )
    result_unbiased = crps_ens(
        pred, target, estimator="unbiased",
        ens_dim=1, average_grid=True, sum_vars=True,
    )

    # almost-fair with alpha=0: diff_factor = (M-1+0)/(M*(M-1)) = 1/M
    # This equals the biased estimator, NOT unbiased.
    # So they should NOT be equal.
    # Almost-fair with alpha=1: diff_factor = (M-1+1)/(M*(M-1)) = 1/(M-1)
    # which equals unbiased.
    result_af_1 = crps_ens(
        pred, target, estimator="almost-fair", afc_alpha=1.0,
        ens_dim=1, average_grid=True, sum_vars=True,
    )

    assert torch.isclose(result_af_1, result_unbiased, atol=1e-5), (
        "Almost-fair CRPS with alpha=1 should equal unbiased CRPS"
    )


