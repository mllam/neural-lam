# Third-party
import torch

# First-party
from neural_lam.metrics import crps_ens, spread_squared


def test_spread_squared():
    # Simulate ensemble prediction: (batch, ensemble_members, nodes, features)
    # Shape: (1, 4, 1, 1). Values: 2, 4, 4, 6
    pred = torch.tensor([[[[2.0]], [[4.0]], [[4.0]], [[6.0]]]])
    target = torch.tensor([[[4.0]]])  # Dummy target
    pred_std = torch.ones_like(pred)  # Dummy pred_std

    # Variance of [2, 4, 4, 6] with Bessel's correction (ddof=1) is 8 / 3 = 2.666...
    # torch.var by default uses unbiased estimator
    result = spread_squared(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )

    expected_variance = torch.tensor([[[2.0, 4.0, 4.0, 6.0]]]).var()
    
    assert torch.isclose(result, expected_variance, atol=1e-5), \
        f"Expected {expected_variance}, got {result}"


def test_crps_ens_small_ensemble():
    # Test crps_ens on a small ensemble (num_ens < 10)
    # Shape: (1, 3, 1, 1) - 3 ensemble members
    pred = torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[3.0]]])  # target is 3
    pred_std = torch.ones_like(pred)
    
    # Calculate expected CRPS manually
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    # X = {1, 2, 5}, y = 3
    # E|X - y| = (2 + 1 + 2) / 3 = 5/3 = 1.666...
    # E|X - X'| difference pairs:
    # |1-1|=0, |1-2|=1, |1-5|=4
    # |2-1|=1, |2-2|=0, |2-5|=3
    # |5-1|=4, |5-2|=3, |5-5|=0
    # Sum = 16. Mean = 16/9 = 1.777...
    # Expected CRPS = 1.666... - 0.5 * 1.777... = 1.666... - 0.888... = 0.777...
    #
    # However, neural-lam uses the unbiased estimator:
    # CRPS_unbiased = mean(|X_i - y|) - (1 / (2 * M * (M-1))) * sum(|X_i - X_j|)
    # Mean absolute error term = 5/3
    # Pairwise diff sum (i != j) = 16
    # Pairwise term = 1 / (2 * 3 * 2) * 16 = 16 / 12 = 4/3 = 1.333...
    # Expected unbiased CRPS = 5/3 - 4/3 = 1/3 = 0.333...

    result = crps_ens(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )

    expected_crps = torch.tensor(1.0 / 3.0)
    
    assert torch.isclose(result, expected_crps, atol=1e-5), \
        f"Expected {expected_crps}, got {result}"


def test_crps_ens_large_ensemble():
    # Test crps_ens on a large ensemble (num_ens >= 10)
    # We will use 10 ensemble members
    pred = torch.arange(10.0).view(1, 10, 1, 1)  # Members 0 to 9
    target = torch.tensor([[[4.5]]])  # Target in the middle
    pred_std = torch.ones_like(pred)

    result = crps_ens(
        pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, ens_dim=1
    )
    
    # For a uniform set {0,1,2,3,4,5,6,7,8,9} and target 4.5
    # MAE = (|0-4.5| + |1-4.5| + ... + |9-4.5|) / 10 = (4.5+3.5+2.5+1.5+0.5)*2 / 10 = 25 / 10 = 2.5
    # Pairwise diff sum: sum_{i!=j} |i-j| = 2 * (1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1) = 330
    # Pair term = 330 / (2 * 10 * 9) = 330 / 180 = 1.833...
    # Unbiased CRPS = 2.5 - 1.833... = 0.666...
    
    expected_crps = torch.tensor(2.5 - (330 / 180))

    assert torch.isclose(result, expected_crps, atol=1e-5), \
        f"Expected {expected_crps}, got {result}"

