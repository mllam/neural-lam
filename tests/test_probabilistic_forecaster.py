# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam import config as nlconfig
from neural_lam import metrics
from neural_lam.models import (
    ARForecaster,
    ForecasterModule,
    ProbabilisticARForecaster,
    ProbabilisticForecasterModule,
    StepPredictor,
)
from tests.conftest import init_datastore_example


class ZeroStepPredictor(StepPredictor):
    """Deterministic predictor always predicting the zero state."""

    def forward(self, prev_state, prev_prev_state, forcing):
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.zeros_like(prev_state) if self.output_std else None
        return pred_state, pred_std


class NoisyStepPredictor(StepPredictor):
    """Stochastic predictor sampling a fresh state at every call."""

    def __init__(self, datastore, **kwargs):
        super().__init__(datastore, **kwargs)
        self.noise_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, prev_state, prev_prev_state, forcing):
        pred_state = self.noise_scale * torch.randn_like(prev_state)
        return pred_state, None


def _example_batch(datastore, B=2, pred_steps=3):
    """Create constant example input tensors matching the datastore dims."""
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, pred_steps, num_grid_nodes, d_forcing)
    target_states = torch.ones(B, pred_steps, num_grid_nodes, d_state) * 5.0
    return init_states, forcing_features, target_states


def test_ar_forecaster_training_loss_matches_direct_score():
    datastore = init_datastore_example("mdp")
    predictor = ZeroStepPredictor(datastore=datastore, output_std=False)
    forecaster = ARForecaster(predictor, datastore, loss="mse")

    init_states, forcing_features, target_states = _example_batch(datastore)
    score_metric = metrics.get_metric("mse")
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)
    d_state = target_states.shape[-1]
    # per_var_std is normally computed from config; override directly since
    # this test only cares about the loss computation, not standardization.
    forecaster.per_var_std = torch.ones(d_state)

    batch_loss, loss_components = forecaster.compute_training_loss(
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool=interior_mask_bool,
    )

    prediction, _ = forecaster(init_states, forcing_features, target_states)
    expected_loss = torch.mean(
        score_metric(
            prediction,
            target_states,
            forecaster.per_var_std,
            mask=interior_mask_bool,
        )
    )

    assert batch_loss.shape == ()
    assert loss_components == {}
    torch.testing.assert_close(batch_loss, expected_loss)


def test_sample_ensemble_shapes_and_member_variability():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    forecaster = ProbabilisticARForecaster(
        predictor, datastore, ensemble_size=2
    )

    # Override masks to test boundary masking behaviour
    forecaster.interior_mask = torch.zeros_like(forecaster.interior_mask)
    forecaster.interior_mask[0, 0] = 1  # One node is interior
    forecaster.boundary_mask = 1 - forecaster.interior_mask

    B, pred_steps, num_members = 2, 3, 4
    init_states, forcing_features, target_states = _example_batch(
        datastore, B=B, pred_steps=pred_steps
    )
    num_grid_nodes = datastore.num_grid_points
    d_state = target_states.shape[-1]

    torch.manual_seed(42)
    ensemble, ensemble_std = forecaster.sample_ensemble(
        init_states,
        forcing_features,
        target_states,
        num_members=num_members,
    )

    assert ensemble.shape == (
        B,
        num_members,
        pred_steps,
        num_grid_nodes,
        d_state,
    )
    assert ensemble_std is None

    # Members carry independent samples on the interior node
    assert not torch.allclose(ensemble[:, 0, :, 0], ensemble[:, 1, :, 0])
    # Boundary nodes are overwritten with the true state in every member
    assert torch.all(ensemble[:, :, :, 1:] == 5.0)

    # Without an explicit member count the configured ensemble_size is used
    default_ensemble, _ = forecaster.sample_ensemble(
        init_states, forcing_features, target_states
    )
    assert default_ensemble.shape[1] == forecaster.ensemble_size


def test_probabilistic_training_loss_gradient_flow():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    forecaster = ProbabilisticARForecaster(
        predictor, datastore, ensemble_size=2, loss="mse"
    )

    init_states, forcing_features, target_states = _example_batch(datastore)
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)
    d_state = target_states.shape[-1]
    # per_var_std is normally computed from config; override directly since
    # this test only cares about the loss computation, not standardization.
    forecaster.per_var_std = torch.ones(d_state)

    torch.manual_seed(42)
    batch_loss, loss_components = forecaster.compute_training_loss(
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool=interior_mask_bool,
    )

    assert batch_loss.shape == ()
    assert loss_components == {}
    assert torch.isfinite(batch_loss)

    batch_loss.backward()
    assert predictor.noise_scale.grad is not None
    assert predictor.noise_scale.grad != 0.0


def test_probabilistic_forecaster_rejects_empty_ensemble():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)

    with pytest.raises(ValueError, match="ensemble_size"):
        ProbabilisticARForecaster(predictor, datastore, ensemble_size=0)


def test_module_training_step_delegates_to_forecaster():
    datastore = init_datastore_example("mdp")
    predictor = ZeroStepPredictor(datastore=datastore, output_std=False)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = ARForecaster(predictor, datastore, config=config, loss="mse")
    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
    )

    init_states, forcing_features, target_states = _example_batch(datastore)
    batch_times = torch.zeros(init_states.shape[0], target_states.shape[1])
    batch = (init_states, target_states, forcing_features, batch_times)

    batch_loss = model.training_step(batch)

    expected_loss, _ = forecaster.compute_training_loss(
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool=model.interior_mask_bool,
    )

    torch.testing.assert_close(batch_loss, expected_loss)


class MemberCountRecordingForecaster(ProbabilisticARForecaster):
    """ProbabilisticARForecaster recording the requested member count."""

    def sample_ensemble(self, *args, **kwargs):
        self.last_num_members = kwargs.get("num_members")
        return super().sample_ensemble(*args, **kwargs)


def test_probabilistic_module_validation_scores_ensemble_mean():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = MemberCountRecordingForecaster(
        predictor, datastore, ensemble_size=2, config=config
    )
    model = ProbabilisticForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        eval_ensemble_size=3,
    )

    B, pred_steps = 2, 3
    init_states, forcing_features, target_states = _example_batch(
        datastore, B=B, pred_steps=pred_steps
    )
    batch_times = torch.zeros(B, pred_steps)
    batch = (init_states, target_states, forcing_features, batch_times)

    torch.manual_seed(42)
    model.validation_step(batch, 0)

    # Validation samples the configured number of evaluation members
    assert forecaster.last_num_members == 3

    # Ensemble-mean MSE entries are collected for epoch-end aggregation
    d_state = target_states.shape[-1]
    (entry_mses,) = model.val_metrics["ens_mse"]
    assert entry_mses.shape == (B, pred_steps, d_state)
    assert torch.all(torch.isfinite(entry_mses))


def test_probabilistic_module_rejects_empty_eval_ensemble():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = ProbabilisticARForecaster(
        predictor, datastore, ensemble_size=2, config=config
    )

    with pytest.raises(ValueError, match="eval_ensemble_size"):
        ProbabilisticForecasterModule(
            forecaster=forecaster,
            config=config,
            datastore=datastore,
            eval_ensemble_size=0,
        )


def test_probabilistic_module_test_step_not_implemented():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = ProbabilisticARForecaster(
        predictor, datastore, ensemble_size=2, config=config
    )
    model = ProbabilisticForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
    )

    init_states, forcing_features, target_states = _example_batch(datastore)
    batch_times = torch.zeros(init_states.shape[0], target_states.shape[1])
    batch = (init_states, target_states, forcing_features, batch_times)

    with pytest.raises(NotImplementedError):
        model.test_step(batch, 0)
