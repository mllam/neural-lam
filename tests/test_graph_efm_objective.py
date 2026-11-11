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

    def embedd_graph(self, batch_size):
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
    expected = expected[:, :, mask].sum((1, 2, 3)).mean()
    torch.testing.assert_close(logs["crps"], expected)
    torch.testing.assert_close(loss, -logs["elbo_likelihood"] + 0.7 * expected)
    assert not any(predictor.prior_flags)
    forecaster.sample_ensemble.assert_called_once_with(
        init, forcing, target, members
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


def test_probabilistic_yaml_round_trip():
    """Optional Graph-EFM settings survive experiment configuration loading."""
    # First-party
    from neural_lam.config import NeuralLAMConfig, ProbabilisticConfig

    config = NeuralLAMConfig.from_yaml("""
    datastores:
      interior:
        kind: mdp
        config_path: danra.yaml
    probabilistic:
      latent_dim: 8
      prior_layers: 1
      encoder_layers: 2
      decoder_layers: 3
      learn_prior: false
      prior_dist: diagonal
      kl_beta: 0.5
      crps_weight: 0.2
      crps_members: 4
      eval_ensemble_size: 6
    """)
    assert config.probabilistic == ProbabilisticConfig(
        latent_dim=8,
        prior_layers=1,
        encoder_layers=2,
        decoder_layers=3,
        learn_prior=False,
        prior_dist="diagonal",
        kl_beta=0.5,
        crps_weight=0.2,
        crps_members=4,
        eval_ensemble_size=6,
    )
    assert NeuralLAMConfig.from_yaml(config.to_yaml()) == config
