"""Exact objective regressions independent of graph construction and data."""

# Standard library
from unittest.mock import Mock

# Third-party
import pytest
import torch
from torch import nn
from torch.distributions import Normal

# First-party
from neural_lam.models.forecasters.graph_efm import GraphEFMForecaster
from tests.dummy_datastore import DummyDatastore


class ObjectivePredictor(nn.Module):
    """Expose known Gaussian distributions for objective tests."""

    predicts_std = True

    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(0.25))
        self.log_std = nn.Parameter(torch.tensor(0.0))
        self.states = []
        self.prior_flags = []
        self.graphs = []
        self.embedding_calls = 0

    def embedd_graph(self, batch_size):
        self.embedding_calls += 1
        return {}

    def posterior_step(
        self, prev, prev_prev, forcing, target, graph, compute_prior=True
    ):
        self.states.append(prev.detach().clone())
        self.prior_flags.append(compute_prior)
        shape = (prev.shape[0], 2, 1)
        posterior = Normal(self.mean.expand(shape), torch.ones(shape))
        prior = (
            Normal(torch.zeros(shape), torch.ones(shape))
            if compute_prior
            else None
        )
        return (
            prior,
            posterior,
            self.mean.expand_as(prev),
            self.log_std.exp().expand_as(prev),
        )

    def forward(self, prev, prev_prev, forcing, graph_emb=None):
        self.graphs.append(graph_emb)
        return self.mean + torch.randn_like(prev), self.log_std.exp().expand_as(
            prev
        )


@pytest.fixture
def objective():
    datastore = DummyDatastore(n_grid_points=25)
    predictor = ObjectivePredictor()
    forecaster = GraphEFMForecaster(predictor, datastore, kl_beta=0.3)
    mask = forecaster.interior_mask[0, :, 0].bool()
    init = torch.zeros(2, 2, 25, 5)
    forcing = torch.zeros(2, 3, 25, 6)
    target = torch.ones(2, 3, 25, 5)
    return forecaster, predictor, mask, init, forcing, target


def test_gaussian_elbo_and_boundary_rollout(objective):
    forecaster, predictor, mask, init, forcing, target = objective
    loss, logs = forecaster.compute_training_loss(init, forcing, target, mask)
    expected_nll = -Normal(torch.tensor(0.25), torch.tensor(1.0)).log_prob(
        torch.tensor(1.0)
    )
    expected_nll = expected_nll * mask.sum() * 5 * 3
    expected_kl = torch.tensor(0.5 * 0.25**2 * 2 * 3)
    torch.testing.assert_close(loss, expected_nll + 0.3 * expected_kl)
    torch.testing.assert_close(logs["elbo_kl"], expected_kl)
    assert len(predictor.states) == 3
    torch.testing.assert_close(
        predictor.states[1][:, ~mask], target[:, 0, ~mask]
    )
    torch.testing.assert_close(
        predictor.states[1][:, mask], torch.full_like(target[:, 0, mask], 0.25)
    )
    loss.backward()
    assert predictor.mean.grad.abs() > 0
    assert predictor.log_std.grad > 0


@pytest.mark.parametrize("members", [2, 5])
def test_crps_matches_pairwise_fair_estimator(objective, members):
    forecaster, predictor, mask, init, forcing, target = objective
    forecaster.kl_beta = 0
    forecaster.crps_weight = 0.7
    forecaster.crps_members = members
    torch.manual_seed(7)
    ensemble = torch.randn(2, members, 3, 25, 5, requires_grad=True)
    forecaster.sample_ensemble = Mock(return_value=(ensemble, None))
    loss, logs = forecaster.compute_training_loss(init, forcing, target, mask)
    first = (ensemble - target.unsqueeze(1)).abs().mean(1)
    pairwise = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs().sum((1, 2))
    expected = first - pairwise / (2 * members * (members - 1))
    expected = expected[:, :, mask].sum(-1).mean()
    torch.testing.assert_close(logs["crps"], expected)
    torch.testing.assert_close(loss, -logs["elbo_likelihood"] + 0.7 * expected)
    assert not any(predictor.prior_flags)
    forecaster.sample_ensemble.assert_called_once_with(
        init, forcing, target, members, graph_emb={}
    )
    loss.backward()
    assert ensemble.grad.abs().sum() > 0


def test_zero_crps_skips_prior_trajectories(objective):
    forecaster, predictor, mask, init, forcing, target = objective
    forecaster.kl_beta = 0
    forecaster.sample_ensemble = Mock(
        side_effect=AssertionError("unexpected sampling")
    )
    forecaster.compute_training_loss(init, forcing, target, mask)
    assert not any(predictor.prior_flags)
    forecaster.sample_ensemble.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kl_beta": -1},
        {"kl_beta": float("nan")},
        {"crps_weight": -1},
        {"crps_weight": float("inf")},
        {"crps_members": 1},
    ],
)
def test_invalid_objective_settings(kwargs):
    with pytest.raises(ValueError, match="must be"):
        GraphEFMForecaster(
            ObjectivePredictor(), DummyDatastore(n_grid_points=25), **kwargs
        )




@pytest.mark.parametrize(
    "devices,num_nodes,distributed",
    [
        (1, 1, False),
        ([0], 1, False),
        (2, 1, True),
        ([0, 1], 1, True),
        (1, 2, True),
        ("auto", 1, True),
    ],
)
@pytest.mark.parametrize("kl_beta", [0, 1])
def test_graph_efm_training_strategy(
    objective, monkeypatch, devices, num_nodes, distributed, kl_beta
):
    """Unused-parameter detection is confined to distributed warm-up."""
    # First-party
    from neural_lam.utils import training_strategy

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    forecaster = objective[0]
    forecaster.kl_beta = kl_beta
    expected = (
        "ddp_find_unused_parameters_true"
        if distributed and kl_beta == 0
        else "auto"
    )
    assert (
        training_strategy(
            devices, num_nodes, find_unused_parameters=forecaster.kl_beta == 0
        )
        == expected
    )


def test_graph_efm_cli_options_and_checkpoint_overrides():
    """Stage changes override saved weights, including explicit zeros."""
    # First-party
    from neural_lam.train_model import build_parser, resolve_graph_efm_args

    parser = build_parser()
    saved = parser.parse_args(
        [
            "--config_path",
            "unused.yaml",
            "--latent_dim",
            "8",
            "--kl_beta",
            "1",
            "--crps_weight",
            "0.2",
            "--no-learn_prior",
        ]
    )
    current = parser.parse_args(
        [
            "--config_path",
            "unused.yaml",
            "--kl_beta",
            "0",
            "--crps_weight",
            "0",
            "--learn_prior",
        ]
    )
    resolved = resolve_graph_efm_args(current, saved)
    assert resolved.latent_dim == 8
    assert resolved.kl_beta == 0
    assert resolved.crps_weight == 0
    assert resolved.learn_prior is True
    assert resolved.crps_members == 2
    assert current.latent_dim is None
