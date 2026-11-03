"""Numerical and reduction checks for reusable fair ensemble CRPS."""

# Third-party
import pytest
import torch

# First-party
from neural_lam.metrics import crps_ens_fair, mask_and_reduce_metric


@pytest.mark.parametrize("members", [2, 3, 5])
@pytest.mark.parametrize("ens_dim", [0, 1, -4])
@pytest.mark.parametrize(
    "average_grid,sum_vars",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_fair_crps_matches_pairwise_reductions(
    members,
    ens_dim,
    average_grid,
    sum_vars,
    masked,
):
    """Sorted scoring matches pairwise scoring for every supported reduction."""
    torch.manual_seed(6)
    ensemble = torch.randn(2, members, 3, 4, 2, dtype=torch.float64)
    target = torch.randn(2, 3, 4, 2, dtype=torch.float64)
    mask = torch.tensor([True, False, True, False]) if masked else None
    first = (ensemble - target.unsqueeze(1)).abs().mean(dim=1)
    pairs = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs().sum((1, 2))
    expected = first - pairs / (2 * members * (members - 1))
    expected = mask_and_reduce_metric(expected, mask, average_grid, sum_vars)
    actual = crps_ens_fair(
        ensemble.movedim(1, ens_dim),
        target,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
        ens_dim=ens_dim,
    )
    torch.testing.assert_close(actual, expected)


def test_fair_crps_retains_gradients():
    """Ensemble objectives can differentiate through reusable scoring."""
    ensemble = torch.tensor([[[[0.0]], [[2.0]], [[3.0]]]], requires_grad=True)
    target = torch.tensor([[[1.0]]])
    crps_ens_fair(ensemble, target).sum().backward()
    assert torch.isfinite(ensemble.grad).all()
    assert ensemble.grad.abs().sum() > 0


@pytest.mark.parametrize("members", [0, 1])
def test_fair_crps_rejects_insufficient_members(members):
    with pytest.raises(ValueError, match="at least two"):
        crps_ens_fair(torch.zeros(2, members, 4, 3), torch.zeros(2, 4, 3))
