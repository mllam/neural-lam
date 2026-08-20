# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam import config as nlconfig
from neural_lam import metrics
from neural_lam.loss_weighting import get_per_var_std
from neural_lam.models import (
    BaseEnsembleARForecaster,
    BaseForecastingModule,
    DeterministicARForecaster,
    DeterministicForecastingModule,
    EnsembleForecastingModule,
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


class ConcreteEnsembleARForecaster(BaseEnsembleARForecaster):
    """
    Test-only concrete ``BaseEnsembleARForecaster``.

    ``BaseEnsembleARForecaster`` supplies no training objective (no single
    default fits every stochastic model), so tests that only need a working
    forecaster to instantiate use this example ensemble-mean objective
    rather than the base class directly. Being the concrete class, it also
    owns whatever that objective needs: the scoring rule, the constant
    per-variable std, and the member count to train on (``forward`` always
    requires an explicit one).
    """

    def __init__(
        self,
        predictor,
        datastore,
        config=None,
        loss: str = "wmse",
        train_num_members: int = 2,
    ):
        super().__init__(predictor, datastore)
        self.loss = metrics.get_metric(loss)
        per_var_std = (
            get_per_var_std(config=config, datastore=datastore)
            if config is not None
            else None
        )
        # Buffer rather than plain attribute, as
        # BaseDeterministicForecaster does, so it follows the module onto
        # the accelerator
        self.register_buffer("per_var_std", per_var_std, persistent=False)
        self.train_num_members = train_num_members

    def compute_training_loss(
        self,
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool,
    ):
        ensemble, per_member_std = self(
            init_states,
            forcing_features,
            target_states,
            num_members=self.train_num_members,
        )
        ensemble_mean = ensemble.mean(dim=1)
        pred_std = (
            per_member_std.mean(dim=1)
            if per_member_std is not None
            else self.per_var_std
        )
        batch_loss = torch.mean(
            self.loss(
                ensemble_mean,
                target_states,
                pred_std,
                mask=interior_mask_bool,
            )
        )
        return batch_loss, {}


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
    forecaster = DeterministicARForecaster(predictor, datastore, loss="mse")

    init_states, forcing_features, target_states = _example_batch(datastore)
    score_metric = metrics.get_metric("mse")
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)

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
            mask=interior_mask_bool,
        )
    )

    assert batch_loss.shape == ()
    assert loss_components == {}
    torch.testing.assert_close(batch_loss, expected_loss)


def test_ensemble_forward_shapes_and_member_variability():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    forecaster = ConcreteEnsembleARForecaster(predictor, datastore)

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
    ensemble, per_member_std = forecaster(
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
    assert per_member_std is None

    # Members carry independent samples on the interior node
    assert not torch.allclose(ensemble[:, 0, :, 0], ensemble[:, 1, :, 0])
    # Boundary nodes are overwritten with the true state in every member
    assert torch.all(ensemble[:, :, :, 1:] == 5.0)


def test_ensemble_training_loss_gradient_flow():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    forecaster = ConcreteEnsembleARForecaster(
        predictor, datastore, loss="mse", train_num_members=2
    )

    init_states, forcing_features, target_states = _example_batch(datastore)
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)
    d_state = target_states.shape[-1]
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


def test_ensemble_forward_rejects_empty_member_count():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    forecaster = ConcreteEnsembleARForecaster(predictor, datastore)
    init_states, forcing_features, target_states = _example_batch(datastore)

    with pytest.raises(ValueError, match="num_members"):
        forecaster(init_states, forcing_features, target_states, num_members=0)


def test_ensemble_ar_forecaster_is_abstract():
    """BaseEnsembleARForecaster leaves compute_training_loss abstract, so
    it cannot be instantiated directly; only a subclass that defines an
    objective can."""
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)

    with pytest.raises(TypeError, match="abstract"):
        BaseEnsembleARForecaster(predictor, datastore)


def test_saved_hparams_hold_resolved_values():
    """Saved hyperparameters are the values the module runs on.

    ``save_hyperparameters`` defaults to inspecting the constructor chain
    and recording the arguments of the most derived ``__init__``, which for
    a subclass with its own signature are the values as passed, before
    ``BaseForecastingModule`` resolves them. ``hparams_initial`` is asserted
    alongside ``hparams`` because it is snapshotted inside
    ``save_hyperparameters``: any attempt to correct the values after that
    call returns reaches only ``hparams`` and would fail here.
    """
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = ZeroStepPredictor(datastore=datastore, output_std=False)
    module = DeterministicForecastingModule(
        forecaster=DeterministicARForecaster(
            predictor, datastore, config=config
        ),
        config=config,
        datastore=datastore,
    )

    # Defaults resolved by the base class, not the None it was called with
    assert module.hparams.val_steps_to_log == [1]
    assert module.hparams.train_steps_to_log == []
    assert module.hparams.metrics_watch == []
    assert module.hparams.var_leads_metrics_watch == {}
    assert dict(module.hparams_initial) == dict(module.hparams)

    # The forecaster and datastore are supplied on load, never saved
    assert "forecaster" not in module.hparams
    assert "datastore" not in module.hparams

    # A subclass's own hyperparameters survive alongside the base's
    ens_module = EnsembleForecastingModule(
        forecaster=ConcreteEnsembleARForecaster(
            NoisyStepPredictor(datastore=datastore, output_std=False),
            datastore,
            config=config,
        ),
        config=config,
        datastore=datastore,
        eval_ensemble_size=3,
    )
    assert ens_module.hparams.eval_ensemble_size == 3
    assert ens_module.hparams.val_steps_to_log == [1]
    assert dict(ens_module.hparams_initial) == dict(ens_module.hparams)


def test_forecasting_module_evaluation_steps_are_abstract():
    """A module omitting validation_step/test_step fails at construction.

    LightningModule defines both as no-op stubs rather than abstract
    methods, so without BaseForecastingModule declaring them abstract such a
    module would instantiate happily and silently skip evaluation.
    """

    class MissingEvaluationSteps(BaseForecastingModule):
        pass

    assert {
        "validation_step",
        "test_step",
    } <= MissingEvaluationSteps.__abstractmethods__

    with pytest.raises(TypeError, match="validation_step|test_step"):
        MissingEvaluationSteps(forecaster=None, config=None, datastore=None)


def test_module_training_step_delegates_to_forecaster():
    datastore = init_datastore_example("mdp")
    predictor = ZeroStepPredictor(datastore=datastore, output_std=False)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = DeterministicARForecaster(
        predictor, datastore, config=config, loss="mse"
    )
    model = DeterministicForecastingModule(
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


def test_deterministic_training_step_logs_per_step_losses():
    """--train_steps_to_log selects per-step training losses to report,
    which the deterministic objective can supply because it decomposes
    over rollout steps."""
    datastore = init_datastore_example("mdp")
    predictor = ZeroStepPredictor(datastore=datastore, output_std=False)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = DeterministicARForecaster(
        predictor, datastore, config=config, loss="mse"
    )
    model = DeterministicForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        train_steps_to_log=[1, 3],
    )

    pred_steps = 3
    init_states, forcing_features, target_states = _example_batch(
        datastore, pred_steps=pred_steps
    )
    batch_times = torch.zeros(init_states.shape[0], pred_steps)
    batch = (init_states, target_states, forcing_features, batch_times)

    captured = {}
    model.log_dict = lambda log_dict, **kwargs: captured.update(log_dict)

    batch_loss = model.training_step(batch)

    assert set(captured) == {
        "train_loss",
        "train_loss_unroll1",
        "train_loss_unroll3",
    }
    torch.testing.assert_close(captured["train_loss"], batch_loss)

    # The reported steps are the corresponding entries of the same
    # decomposition the logged train_loss averages
    prediction, pred_std = forecaster(
        init_states, forcing_features, target_states
    )
    step_losses = torch.mean(
        forecaster.compute_loss_from_forecast(
            prediction,
            target_states,
            pred_std,
            mask=model.interior_mask_bool,
        ),
        dim=0,
    )
    torch.testing.assert_close(captured["train_loss_unroll1"], step_losses[0])
    torch.testing.assert_close(captured["train_loss_unroll3"], step_losses[2])
    torch.testing.assert_close(batch_loss, torch.mean(step_losses))


class MemberCountRecordingForecaster(ConcreteEnsembleARForecaster):
    """Ensemble forecaster recording the requested member count."""

    def forward(self, *args, **kwargs):
        self.last_num_members = kwargs.get("num_members")
        return super().forward(*args, **kwargs)


class ComponentReportingForecaster(MemberCountRecordingForecaster):
    """Forecaster splitting its objective into named loss components.

    Records every requested member count in order, so that the objective's
    own ensemble can be told apart from the evaluation one.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the forecaster and its recording state."""
        super().__init__(*args, **kwargs)
        self.num_members_seen: list[int] = []
        self.last_batch_loss = None
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(self, *args, **kwargs):
        self.num_members_seen.append(kwargs.get("num_members"))
        return super().forward(*args, **kwargs)

    def compute_training_loss(self, *args, **kwargs):
        batch_loss, _ = super().compute_training_loss(*args, **kwargs)
        self.last_batch_loss = batch_loss
        self.last_components = {"kl": torch.tensor(0.5)}
        return batch_loss, self.last_components


def test_ensemble_module_validation_scores_ensemble_mean():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = MemberCountRecordingForecaster(
        predictor, datastore, config=config
    )
    model = EnsembleForecastingModule(
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


def test_ensemble_module_rejects_empty_eval_ensemble():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = ConcreteEnsembleARForecaster(
        predictor, datastore, config=config
    )

    with pytest.raises(ValueError, match="eval_ensemble_size"):
        EnsembleForecastingModule(
            forecaster=forecaster,
            config=config,
            datastore=datastore,
            eval_ensemble_size=0,
        )


def test_ensemble_module_test_step_scores_ensemble_mean():
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = MemberCountRecordingForecaster(
        predictor, datastore, config=config
    )
    model = EnsembleForecastingModule(
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
    model.test_step(batch, 0)

    # Test samples the configured number of evaluation members
    assert forecaster.last_num_members == 3

    # Ensemble-mean MSE entries are collected for epoch-end aggregation
    d_state = target_states.shape[-1]
    (entry_mses,) = model.test_metrics["ens_mse"]
    assert entry_mses.shape == (B, pred_steps, d_state)
    assert torch.all(torch.isfinite(entry_mses))


def test_ensemble_module_logs_forecaster_objective():
    """Validation reports the forecaster's own training objective, giving
    ModelCheckpoint a val_mean_loss to monitor. Testing does not: nothing
    monitors it there, so the extra forward pass would buy nothing."""
    datastore = init_datastore_example("mdp")
    predictor = NoisyStepPredictor(datastore=datastore, output_std=False)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = ComponentReportingForecaster(
        predictor, datastore, config=config, train_num_members=2
    )
    model = EnsembleForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        eval_ensemble_size=3,
    )

    B, pred_steps = 2, 3
    init_states, forcing_features, target_states = _example_batch(
        datastore, B=B, pred_steps=pred_steps
    )
    batch = (
        init_states,
        target_states,
        forcing_features,
        torch.zeros(B, pred_steps),
    )

    captured = {}
    model.log_dict = lambda log_dict, **kwargs: captured.update(log_dict)

    torch.manual_seed(42)
    model.validation_step(batch, 0)

    # The objective is the forecaster's own, reported under the same name
    # the deterministic module uses, alongside any components it splits out
    assert captured["val_mean_loss"] is forecaster.last_batch_loss
    assert captured["val_kl"] is forecaster.last_components["kl"]

    # The objective samples its own ensemble, with the member count it
    # trains on, before the evaluation ensemble is drawn
    assert forecaster.num_members_seen == [2, 3]

    # Testing draws only the evaluation ensemble, and logs no loss
    captured.clear()
    forecaster.num_members_seen.clear()
    model.test_step(batch, 0)

    assert forecaster.num_members_seen == [3]
    assert captured == {}
