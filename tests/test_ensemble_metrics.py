"""Numerical and reduction checks for reusable fair ensemble CRPS."""

# Third-party
import pytest
import torch

# First-party
from neural_lam.metrics import crps_ens, crps_ens_fair, mask_and_reduce_metric


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


@pytest.mark.parametrize("members", [2, 5])
@pytest.mark.parametrize(
    "estimator,alpha",
    [
        ("biased", None),
        ("unbiased", None),
        ("almost-fair", 0.0),
        ("almost-fair", 0.4),
        ("almost-fair", 1.0),
    ],
)
def test_general_crps_estimators(members, estimator, alpha):
    """Every estimator agrees with its pairwise definition and gradients."""
    torch.manual_seed(8)
    ensemble = torch.randn(2, members, 3, 2, requires_grad=True)
    target = torch.randn(2, 3, 2)
    first = (ensemble - target.unsqueeze(1)).abs().mean(1)
    pairs = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs().sum((1, 2))
    factor = {
        "biased": 1 / members,
        "unbiased": 1 / (members - 1),
        "almost-fair": (members - 1 + (alpha or 0)) / (members * (members - 1)),
    }[estimator]
    expected = first - pairs * factor / (2 * members)
    actual = crps_ens(
        ensemble,
        target,
        average_grid=False,
        sum_vars=False,
        estimator=estimator,
        afc_alpha=alpha,
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(ensemble.grad).all()


@pytest.mark.parametrize("alpha", [None, -1, 2, float("nan"), float("inf")])
def test_almost_fair_crps_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="afc_alpha"):
        crps_ens(
            torch.zeros(1, 2, 3, 2),
            torch.zeros(1, 3, 2),
            estimator="almost-fair",
            afc_alpha=alpha,
        )


def test_empirical_crps_single_member_is_mae():
    ensemble = torch.tensor([[[[1.0, 4.0]]]])
    target = torch.tensor([[[2.0, 2.0]]])
    torch.testing.assert_close(
        crps_ens(ensemble, target, estimator="biased"), torch.tensor([3.0])
    )


def test_unknown_crps_estimator():
    with pytest.raises(ValueError, match="Unknown CRPS estimator"):
        crps_ens(torch.zeros(1, 2, 3, 2), torch.zeros(1, 3, 2), estimator="bad")
