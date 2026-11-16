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
    with pytest.warns(UserWarning, match="single ensemble member"):
        actual = crps_ens(ensemble, target, estimator="biased")
    torch.testing.assert_close(actual, torch.tensor([3.0]))


def test_unknown_crps_estimator():
    with pytest.raises(ValueError, match="Unknown CRPS estimator"):
        crps_ens(torch.zeros(1, 2, 3, 2), torch.zeros(1, 3, 2), estimator="bad")


@pytest.mark.parametrize(
    "metric_name,kwargs",
    [
        ("crps_ens_fair", {"estimator": "unbiased"}),
        ("crps_ens_biased", {"estimator": "biased"}),
        (
            "crps_ens_almost_fair",
            {"estimator": "almost-fair", "afc_alpha": 0.95},
        ),
    ],
)
def test_registered_crps_variants_follow_shared_metric_api(metric_name, kwargs):
    # First-party
    from neural_lam.metrics import (
        get_metric,
        requires_ensemble,
        requires_pred_std,
    )

    ensemble = torch.tensor([[[[0.0, 1.0]], [[2.0, 4.0]], [[6.0, 9.0]]]])
    target = torch.tensor([[[1.0, 3.0]]])
    metric = get_metric(metric_name)
    assert requires_ensemble(metric)
    assert not requires_pred_std(metric)
    torch.testing.assert_close(
        metric(
            pred=ensemble,
            target=target,
            pred_std=torch.tensor([0.1, 2.0]),
            sum_vars=False,
        ),
        crps_ens(ensemble, target, sum_vars=False, **kwargs),
    )


@pytest.mark.parametrize("sum_vars", [False, True])
@pytest.mark.parametrize("estimator", ["unbiased", "biased", "almost-fair"])
def test_single_member_crps_warns_and_preserves_mask_and_variables(
    sum_vars, estimator
):
    ensemble = torch.arange(12.0).reshape(1, 1, 2, 3, 2)
    target = torch.ones(1, 2, 3, 2)
    mask = torch.tensor([True, False, True])
    with pytest.warns(UserWarning, match="single ensemble member"):
        actual = crps_ens(
            ensemble,
            target,
            torch.tensor([1.0, 2.0]),
            mask=mask,
            sum_vars=sum_vars,
            estimator=estimator,
            afc_alpha=0.95,
        )
    expected = mask_and_reduce_metric(
        (ensemble[:, 0] - target).abs(), mask, True, sum_vars
    )
    torch.testing.assert_close(actual, expected)


def test_deterministic_forecaster_rejects_ensemble_loss():
    # First-party
    from neural_lam.models import DeterministicARForecaster
    from tests.dummy_datastore import DummyDatastore

    with pytest.raises(ValueError, match="requires ensemble predictions"):
        DeterministicARForecaster(
            torch.nn.Identity(),
            DummyDatastore(n_grid_points=25),
            loss="crps_ens",
        )


@pytest.mark.parametrize("members", [2, 5, 12])
@pytest.mark.parametrize("chunk_dim", [0, 2, -2])
@pytest.mark.parametrize(
    "estimator,alpha",
    [("unbiased", None), ("biased", None), ("almost-fair", 0.95)],
)
def test_chunked_crps_matches_values_and_gradients(
    members, chunk_dim, estimator, alpha
):
    torch.manual_seed(12)
    ensemble = torch.randn(
        3, members, 4, 5, 2, dtype=torch.float64, requires_grad=True
    )
    target = torch.randn(3, 4, 5, 2, dtype=torch.float64)
    mask = torch.tensor([True, False, True, False, True])
    kwargs = dict(
        mask=mask,
        average_grid=True,
        sum_vars=False,
        estimator=estimator,
        afc_alpha=alpha,
    )
    expected = crps_ens(ensemble, target, **kwargs)
    actual = crps_ens(
        ensemble, target, chunk_size=2, chunk_dim=chunk_dim, **kwargs
    )
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.sum(), ensemble)[0]
    actual_grad = torch.autograd.grad(actual.sum(), ensemble)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 1, "chunk_dim": 1},
        {"chunk_size": 1, "chunk_dim": 5},
    ],
)
def test_crps_rejects_invalid_chunks(kwargs):
    with pytest.raises(ValueError, match="chunk_"):
        crps_ens(torch.zeros(2, 3, 4, 5, 2), torch.zeros(2, 4, 5, 2), **kwargs)


def test_fair_crps_chunking_supports_reordered_ensemble_axis():
    ensemble = torch.randn(3, 2, 4, 5, 2)
    target = torch.randn(2, 4, 5, 2)
    expected = crps_ens_fair(ensemble, target, ens_dim=0)
    actual = crps_ens_fair(
        ensemble, target, ens_dim=0, chunk_size=2, chunk_dim=2
    )
    torch.testing.assert_close(actual, expected)
