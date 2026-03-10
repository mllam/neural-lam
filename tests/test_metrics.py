# Third-party
import torch

# First-party
from neural_lam import metrics


def test_weighted_grid_average_with_mask():
    pred = torch.tensor([[[[2.0], [4.0], [6.0]]]])  # (B=1, T=1, N=3, d=1)
    target = torch.zeros_like(pred)
    pred_std = torch.ones(1)  # ignored by mse

    mask = torch.tensor([True, False, True])
    grid_weights = torch.tensor([1.0, 100.0, 3.0])

    out = metrics.mse(
        pred,
        target,
        pred_std,
        mask=mask,
        average_grid=True,
        sum_vars=False,
        grid_weights=grid_weights,
    )

    # Keep indices 0 and 2 due to mask; normalized weights are [1/4, 3/4].
    # Weighted MSE is (2^2)*(1/4) + (6^2)*(3/4) = 28.
    expected = torch.tensor([[[28.0]]])
    assert torch.allclose(out, expected)


def test_grid_weights_ignored_without_grid_reduction():
    pred = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    target = torch.zeros_like(pred)
    pred_std = torch.ones(1)
    grid_weights = torch.tensor([1.0, 2.0, 3.0])

    weighted = metrics.mse(
        pred,
        target,
        pred_std,
        average_grid=False,
        sum_vars=False,
        grid_weights=grid_weights,
    )
    unweighted = metrics.mse(
        pred,
        target,
        pred_std,
        average_grid=False,
        sum_vars=False,
    )

    assert torch.allclose(weighted, unweighted)
