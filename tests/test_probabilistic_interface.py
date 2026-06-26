"""Interface-level checks for the latent-variable forecasting stack.

Drives a minimal, graph-free dummy ``LatentStepPredictor`` through
``ProbabilisticARForecaster`` and ``ProbabilisticForecasterModule`` to verify
the predictor -> forecaster -> module -> ELBO contract independently of any
concrete predictor implementation. The Graph-EFM wiring is tested separately.
"""

# Standard library
import warnings

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb
from torch import nn

# First-party
from neural_lam import config as nlconfig
from neural_lam.models import (
    LatentStepPredictor,
    ProbabilisticARForecaster,
    ProbabilisticForecasterModule,
)
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example

NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1
NUM_MESH_NODES = 3
LATENT_DIM = 4


class _DummyLatentPredictor(LatentStepPredictor):
    """Minimal graph-free latent predictor exercising the interface."""

    def __init__(self, datastore):
        """
        Build a graph-free latent predictor over ``datastore``.

        Parameters
        ----------
        datastore : BaseDatastore
            Datastore providing grid metadata and state-variable counts.
        """
        super().__init__(datastore=datastore, output_std=False)
        d_state = datastore.get_num_data_vars(category="state")
        self.encode_latent = nn.Linear(d_state, LATENT_DIM)
        self.decode_latent = nn.Linear(LATENT_DIM, d_state)

    def _latent_dist(self, state):
        """
        Build a diagonal-Normal latent distribution from a grid state.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. State summarised
            into latent parameters by mean-pooling over the grid.

        Returns
        -------
        torch.distributions.Normal
            Shape ``(B, NUM_MESH_NODES, LATENT_DIM)``. Unit-variance Normal.
        """
        mean = self.encode_latent(state.mean(dim=1))  # (B, LATENT_DIM)
        mean = mean.unsqueeze(1).expand(-1, NUM_MESH_NODES, -1)
        return torch.distributions.Normal(mean, torch.ones_like(mean))

    def step_distributions(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        target=None,
        compute_prior=True,
    ):
        """
        Latent distributions and reconstruction for one step.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_{t-1}``;
            unused by this dummy.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, num_forcing_vars)``; unused.
        target : torch.Tensor or None, optional
            Shape ``(B, num_grid_nodes, num_state_vars)``. True ``X_{t+1}``.
        compute_prior : bool, optional
            Whether to build the prior on the training path.

        Returns
        -------
        prior_dist : torch.distributions.Normal or None
            Prior over the latent, or ``None`` when skipped.
        posterior_dist : torch.distributions.Normal or None
            Posterior over the latent, or ``None`` on the eval path.
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. Decoded mean.
        pred_std : None
            This dummy never emits an std.
        """
        if target is not None:
            posterior_dist = self._latent_dist(target)
            latent = posterior_dist.rsample()
            prior_dist = (
                self._latent_dist(prev_state) if compute_prior else None
            )
        else:
            posterior_dist = None
            prior_dist = self._latent_dist(prev_state)
            latent = prior_dist.rsample()

        delta = self.decode_latent(latent.mean(dim=1)).unsqueeze(1)
        pred_mean = prev_state + delta
        return prior_dist, posterior_dist, pred_mean, None


def _build_module(kl_beta=1.0):
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = _DummyLatentPredictor(datastore)
    forecaster = ProbabilisticARForecaster(predictor, datastore)
    module = ProbabilisticForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="nll",
        kl_beta=kl_beta,
    )
    return datastore, module


def _make_batch(datastore, pred_steps=3, batch_size=2):
    num_grid = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    init_states = torch.randn(batch_size, 2, num_grid, d_state)
    target_states = torch.randn(batch_size, pred_steps, num_grid, d_state)
    forcing = torch.randn(batch_size, pred_steps, num_grid, d_forcing)
    return init_states, target_states, forcing


@pytest.mark.parametrize("kl_beta", [1.0, 0.0])
def test_training_rollout_terms(kl_beta):
    """training_rollout returns per-step means/kl; kl is None when β == 0."""
    datastore, module = _build_module(kl_beta=kl_beta)
    init_states, target_states, forcing = _make_batch(datastore)

    pred_means, pred_stds, kl_terms = module.forecaster.training_rollout(
        init_states,
        forcing,
        target_states,
        compute_kl=kl_beta > 0,
    )
    b, t = init_states.shape[0], forcing.shape[1]
    assert pred_means.shape == (
        b,
        t,
        datastore.num_grid_points,
        datastore.get_num_data_vars(category="state"),
    )
    assert pred_stds is None  # output_std=False
    if kl_beta > 0:
        assert kl_terms.shape == (b, t)
    else:
        assert kl_terms is None


def test_sample_trajectories_shape():
    """sample_trajectories stacks an ensemble dimension of prior rollouts."""
    datastore, module = _build_module()
    init_states, target_states, forcing = _make_batch(datastore)

    num_traj = 4
    traj_means, traj_stds = module.forecaster.sample_trajectories(
        init_states, forcing, target_states, num_traj
    )
    b, t = init_states.shape[0], forcing.shape[1]
    assert traj_means.shape == (
        b,
        num_traj,
        t,
        datastore.num_grid_points,
        datastore.get_num_data_vars(category="state"),
    )
    assert traj_stds is None  # output_std=False


def test_elbo_loss_is_finite_and_differentiable():
    """The assembled ELBO loss is finite and gradients reach the predictor."""
    datastore, module = _build_module()
    init_states, target_states, forcing = _make_batch(datastore)

    pred_means, _, kl_terms = module.forecaster.training_rollout(
        init_states,
        forcing,
        target_states,
        compute_kl=True,
    )
    entry_log_lik = -module.loss(
        pred_means,
        target_states,
        module.per_var_std,
        mask=module.interior_mask_bool,
        average_grid=False,
        sum_vars=False,
    )
    likelihood = entry_log_lik.sum(dim=(2, 3)).sum(dim=1)  # (B,)
    loss = -likelihood.mean() + module.kl_beta * kl_terms.sum(dim=1).mean()

    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in module.forecaster.predictor.parameters()
    )


def test_end_to_end_probabilistic_training_runs():
    """A full Trainer epoch (train + val) runs through the whole pipeline."""
    datastore, module = _build_module()

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        logger=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wandb.init(mode="disabled")
        trainer.fit(module, data_module)
