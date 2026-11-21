# Standard library
import shutil
import typing
from argparse import Namespace
from typing import TYPE_CHECKING

# Third-party
import pytest
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore.base import BaseDatastore
from neural_lam.models import (
    BaseDeterministicForecaster,
    BaseProbabilisticARForecaster,
    BaseProbabilisticForecaster,
    DeterministicARForecaster,
    DeterministicForecastingModule,
    GraphLAM,
    StepPredictor,
    unroll_forecast,
)
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore

if TYPE_CHECKING:
    # First-party
    from neural_lam.datastore.base import BaseDatastore


class NoStaticDummyDatastore(DummyDatastore):
    """DummyDatastore variant that returns None for static features."""

    def get_dataarray(self, category, split, standardize=False):
        if category == "static":
            return None
        return super().get_dataarray(category, split, standardize=standardize)


class MockStepPredictor(StepPredictor):
    def __init__(self, datastore, **kwargs):
        super().__init__(datastore, **kwargs)

    def forward(self, prev_state, prev_prev_state, forcing):
        # Return zeros for state
        # The true state will be mixed in at boundaries
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.zeros_like(prev_state) if self.output_std else None
        return pred_state, pred_std


@pytest.mark.requires_real_data
def test_ar_forecaster_unroll():
    datastore = init_datastore_example("mdp")
    predictor = MockStepPredictor(
        datastore=datastore,
        output_std=False,
    )

    forecaster = DeterministicARForecaster(predictor, datastore)

    # Override masks to test boundary masking behaviour
    forecaster.interior_mask = torch.zeros_like(forecaster.interior_mask)
    forecaster.interior_mask[0, 0] = 1  # One node is interior
    forecaster.boundary_mask = 1 - forecaster.interior_mask

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    pred_steps = 3
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, pred_steps, num_grid_nodes, d_forcing)
    true_states = torch.ones(B, pred_steps, num_grid_nodes, d_state) * 5.0

    prediction, pred_std = forecaster(
        init_states, forcing_features, true_states
    )

    assert prediction.shape == (B, pred_steps, num_grid_nodes, d_state)

    # Boundary (where interior_mask == 0) should equal true_state (5.0)
    # Interior (where interior_mask == 1) should equal predictor output (0.0)
    assert torch.all(prediction[:, :, 0, :] == 0.0)
    assert torch.all(prediction[:, :, 1:, :] == 5.0)


@pytest.mark.requires_real_data
def test_ar_forecaster_compute_loss_from_forecast():
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        }
    )
    predictor = MockStepPredictor(datastore=datastore, output_std=False)
    forecaster = DeterministicARForecaster(
        predictor, datastore, config=config, loss="mse"
    )

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    prediction = torch.zeros(B, num_grid_nodes, d_state)
    target = torch.ones(B, num_grid_nodes, d_state)
    mask = torch.ones(num_grid_nodes, dtype=torch.bool)

    # pred_std=None falls back to forecaster.per_var_std and applies the
    # forecaster's own configured scoring rule (self.loss)
    scored = forecaster.compute_loss_from_forecast(
        prediction, target, None, mask=mask
    )
    expected = forecaster.loss(
        prediction, target, forecaster.per_var_std, mask=mask
    )
    assert torch.equal(scored, expected)

    # An explicit pred_std is used as-is, not overridden by per_var_std
    explicit_std = torch.full((d_state,), 2.0)
    scored_explicit = forecaster.compute_loss_from_forecast(
        prediction, target, explicit_std, mask=mask
    )
    expected_explicit = forecaster.loss(
        prediction, target, explicit_std, mask=mask
    )
    assert torch.equal(scored_explicit, expected_explicit)


@pytest.mark.requires_real_data
def test_ar_forecaster_without_config_raises_on_use_not_construction():
    """A predictor that doesn't output std plus no config is a valid,
    unambiguous state at construction time (the forecaster may only ever
    be used for inference), so DeterministicARForecaster must not raise
    there. It should only raise once scoring is actually attempted and has
    no std to use, and the error should come from the forecaster itself,
    not a wrapping module."""
    datastore = init_datastore_example("mdp")
    predictor = MockStepPredictor(datastore=datastore, output_std=False)

    # Construction succeeds even though predicts_std=False and config=None
    forecaster = DeterministicARForecaster(predictor, datastore)
    assert forecaster.per_var_std is None

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    prediction = torch.zeros(B, num_grid_nodes, d_state)
    target = torch.ones(B, num_grid_nodes, d_state)

    with pytest.raises(ValueError, match="per_var_std fallback"):
        forecaster.compute_loss_from_forecast(prediction, target, None)

    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    pred_steps = 3
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, pred_steps, num_grid_nodes, d_forcing)
    true_states = torch.ones(B, pred_steps, num_grid_nodes, d_state)

    with pytest.raises(ValueError, match="per_var_std fallback"):
        forecaster.compute_training_loss(
            init_states,
            forcing_features,
            true_states,
            interior_mask_bool=torch.ones(num_grid_nodes, dtype=torch.bool),
        )


@pytest.mark.requires_real_data
def test_ar_forecaster_without_config_scores_with_unweighted_loss():
    """The std fallback is only needed by scoring rules that use a std, so
    an unweighted loss must score without a config, rather than demanding a
    per_var_std it would immediately ignore."""
    datastore = init_datastore_example("mdp")
    predictor = MockStepPredictor(datastore=datastore, output_std=False)

    forecaster = DeterministicARForecaster(predictor, datastore, loss="mse")
    assert forecaster.per_var_std is None

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    prediction = torch.zeros(B, num_grid_nodes, d_state)
    target = torch.ones(B, num_grid_nodes, d_state)

    scored = forecaster.compute_loss_from_forecast(prediction, target, None)
    torch.testing.assert_close(scored, torch.full((B,), float(d_state)))


def test_forecaster_hierarchy_is_a_tree():
    """The two forecaster families share auto-regressive unrolling through
    ``unroll_forecast`` rather than through a common ancestor, so every
    forecaster has a single base and no diamond can arise. ``BaseForecaster``
    itself is the exception, combining nn.Module with ABC."""
    for forecaster_class in (
        BaseDeterministicForecaster,
        DeterministicARForecaster,
        BaseProbabilisticForecaster,
        BaseProbabilisticARForecaster,
    ):
        assert len(forecaster_class.__bases__) == 1, forecaster_class


@pytest.mark.requires_real_data
def test_unroll_forecast_reproduces_deterministic_forward():
    """The unrolling both families call is a plain function, so calling it
    directly gives exactly what DeterministicARForecaster.forward does."""
    datastore = init_datastore_example("mdp")
    predictor = MockStepPredictor(datastore=datastore, output_std=False)
    forecaster = DeterministicARForecaster(predictor, datastore)

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * 3
    pred_steps = 3
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, pred_steps, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, pred_steps, num_grid_nodes, d_state) * 5.0

    prediction, pred_std = forecaster(
        init_states, forcing_features, boundary_states
    )
    direct_prediction, direct_std = unroll_forecast(
        predictor,
        init_states,
        forcing_features,
        boundary_states,
        forecaster.boundary_mask,
        forecaster.interior_mask,
    )

    assert torch.equal(prediction, direct_prediction)
    assert pred_std is None
    assert direct_std is None


@pytest.mark.requires_real_data
def test_forecasting_module_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        }
    )

    # Build predictor and forecaster externally, then inject into
    # DeterministicForecastingModule
    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = DeterministicARForecaster(
        predictor, datastore, config=config, loss="mse"
    )

    model = DeterministicForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    ckpt_path = tmp_path / "test.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    # Build a fresh forecaster structure for loading weights into
    load_predictor = predictor_class(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    load_forecaster = DeterministicARForecaster(
        load_predictor, datastore, config=config, loss="mse"
    )

    # Load from checkpoint
    loaded_model = DeterministicForecastingModule.load_from_checkpoint(
        ckpt_path,
        datastore=datastore,
        forecaster=load_forecaster,
        weights_only=False,
    )

    # Validate the correct internal hierarchy has been constructed
    assert loaded_model.forecaster.predictor.__class__.__name__ == "GraphLAM"

    # Verify that outputs match (checkpoint successfully restored weights)
    B, num_grid_nodes = 2, model.forecaster.predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, 1, num_grid_nodes, d_state) * 5.0

    with torch.no_grad():
        out_before = model.forecaster(
            init_states, forcing_features, boundary_states
        )
        out_after = loaded_model.forecaster(
            init_states, forcing_features, boundary_states
        )

    assert torch.allclose(out_before[0], out_after[0])


@pytest.mark.requires_real_data
def test_forecasting_module_old_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        }
    )

    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    # Use distinctive non-default values so we can detect silent fallback
    # to DeterministicForecastingModule's defaults during load.
    saved_loss = "mse"
    saved_lr = 0.123
    saved_create_gif = True
    saved_val_steps = [2]
    saved_n_example_pred = 7

    forecaster = DeterministicARForecaster(
        predictor, datastore, config=config, loss=saved_loss
    )

    model = DeterministicForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        lr=saved_lr,
        restore_opt=False,
        n_example_pred=saved_n_example_pred,
        create_gif=saved_create_gif,
        val_steps_to_log=saved_val_steps,
        metrics_watch=[],
    )

    ckpt_path = tmp_path / "test_old.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    # Manually hack the checkpoint to emulate a pre-refactor checkpoint:
    # both the state_dict (flat keys) AND the hyper_parameters (nested
    # under an argparse Namespace called 'args' as old ARModel saved them).
    ckpt = torch.load(ckpt_path, weights_only=False)
    old_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("forecaster.predictor."):
            # Revert structural rename to emulate old flat keys
            new_k = k.replace("forecaster.predictor.", "")
            if "encoding_grid_mlp" in new_k:
                new_k = new_k.replace("encoding_grid_mlp", "g2m_gnn.grid_mlp")
            old_state_dict[new_k] = v
        else:
            old_state_dict[k] = v
    ckpt["state_dict"] = old_state_dict

    ckpt["hyper_parameters"] = {
        "args": Namespace(
            loss=saved_loss,
            lr=saved_lr,
            restore_opt=False,
            n_example_pred=saved_n_example_pred,
            create_gif=saved_create_gif,
            val_steps_to_log=saved_val_steps,
            metrics_watch=[],
            var_leads_metrics_watch={},
        ),
        "config": ckpt["hyper_parameters"]["config"],
    }
    torch.save(ckpt, ckpt_path)

    # Build a fresh forecaster structure for loading weights into
    load_predictor = predictor_class(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    load_forecaster = DeterministicARForecaster(
        load_predictor, datastore, config=config, loss=saved_loss
    )

    # Load from hacked old checkpoint
    loaded_model = DeterministicForecastingModule.load_from_checkpoint(
        ckpt_path,
        datastore=datastore,
        forecaster=load_forecaster,
        weights_only=False,
    )

    # Validate the correct internal hierarchy has been constructed
    assert loaded_model.forecaster.predictor.__class__.__name__ == "GraphLAM"

    # Hyperparameters nested in the legacy 'args' namespace must round-trip
    # rather than silently falling back to DeterministicForecastingModule
    # defaults.
    assert loaded_model.hparams.lr == saved_lr
    assert loaded_model.hparams.val_steps_to_log == saved_val_steps
    assert loaded_model.create_gif is saved_create_gif
    assert loaded_model.n_example_pred == saved_n_example_pred

    # Verify that outputs match (checkpoint successfully restored weights)
    B, num_grid_nodes = 2, model.forecaster.predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, 1, num_grid_nodes, d_state) * 5.0

    with torch.no_grad():
        out_before = model.forecaster(
            init_states, forcing_features, boundary_states
        )
        out_after = loaded_model.forecaster(
            init_states, forcing_features, boundary_states
        )

    assert torch.allclose(out_before[0], out_after[0])


@pytest.mark.requires_real_data
def test_graph_lam_no_static_features():
    """GraphLAM (real GNN) should run a forward pass when the datastore has
    no static features - verifying that the empty static tensor flows through
    the graph encoder/processor/decoder without error."""
    base_datastore = init_datastore_example("mdp")

    class NoStaticWrapper:
        """Delegate everything to the real datastore, but return None for
        static so StepPredictor creates an empty (N, 0) static buffer."""

        def __init__(self, ds):
            self._ds = ds

        def __getattr__(self, name):
            return getattr(self._ds, name)

        def get_dataarray(self, category, split=None, standardize=False):
            if category == "static":
                return None
            return self._ds.get_dataarray(
                category, split, standardize=standardize
            )

    datastore = NoStaticWrapper(base_datastore)

    # First-party
    from neural_lam.models import MODELS

    predictor = MODELS["graph_lam"](
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )

    assert predictor.grid_static_features.shape[1] == 0

    forecaster = DeterministicARForecaster(
        predictor, typing.cast("BaseDatastore", datastore)
    )
    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = base_datastore.get_num_data_vars(category="state")
    d_forcing = base_datastore.get_num_data_vars(category="forcing") * 3
    init_states = torch.zeros(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.zeros(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.zeros(B, 1, num_grid_nodes, d_state)

    with torch.no_grad():
        prediction, pred_std = forecaster(
            init_states, forcing_features, boundary_states
        )

    assert prediction.shape == (B, 1, num_grid_nodes, d_state)
    assert pred_std is None


def test_step_predictor_no_static_features():
    """Model should run correctly when the datastore has no static features,
    using an empty (N, 0) tensor in place of static features."""
    datastore = NoStaticDummyDatastore()

    predictor = MockStepPredictor(
        datastore=datastore,
        output_std=False,
    )

    # Static features buffer should exist but be empty (zero width)
    assert predictor.grid_static_features.shape == (
        datastore.num_grid_points,
        0,
    )

    # Verify a forward pass works end-to-end via DeterministicARForecaster
    forecaster = DeterministicARForecaster(predictor, datastore)
    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")
    init_states = torch.zeros(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.zeros(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.zeros(B, 1, num_grid_nodes, d_state)

    prediction, pred_std = forecaster(
        init_states, forcing_features, boundary_states
    )
    assert prediction.shape == (B, 1, num_grid_nodes, d_state)
    assert pred_std is None


@pytest.mark.parametrize("lower,upper", [(5.0, 1.0), (1.0, 1.0)])
def test_step_predictor_invalid_clamping_limits(lower, upper):
    """Lower clamping limit must be smaller than upper, else ValueError.

    Checked with a raise rather than an ``assert``, which ``python -O`` strips.
    """
    datastore = DummyDatastore()
    feature = datastore.get_vars_names(category="state")[0]
    predictor = MockStepPredictor(
        datastore=datastore,
        output_clamping_lower={feature: lower},
        output_clamping_upper={feature: upper},
    )

    with pytest.raises(ValueError, match="must be smaller than upper"):
        predictor.prepare_clamping_params(datastore)


@pytest.fixture(scope="module")
def _dummy_graph_dir(tmp_path_factory):
    """Build the dummy 1level graph once for the test module."""
    graph_dir = tmp_path_factory.mktemp("graph") / "1level"
    create_graph_from_datastore(
        datastore=DummyDatastore(),
        output_root_path=str(graph_dir),
        n_max_levels=1,
    )
    return graph_dir


@pytest.fixture
def dummy_datastore_with_graph(_dummy_graph_dir):
    """Return a DummyDatastore with the pre-built 1level graph."""
    datastore = DummyDatastore()
    shutil.copytree(_dummy_graph_dir, datastore.root_path / "graph" / "1level")
    return datastore


def test_graph_lam_processor_layers_zero_init(dummy_datastore_with_graph):
    """GraphLAM should support processor_layers=0 without crashing."""
    predictor = GraphLAM(
        datastore=dummy_datastore_with_graph,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=0,
    )
    assert predictor.processor_layers == 0
    assert not isinstance(predictor.processor, torch.nn.Module)

    # Test that identity processor passes mesh and edge reps unchanged
    mesh_rep = torch.randn(2, 5, 4)
    edge_rep = torch.randn(2, 10, 4)
    out_mesh, out_edge = predictor.processor(mesh_rep, edge_rep)
    assert torch.equal(out_mesh, mesh_rep)
    assert torch.equal(out_edge, edge_rep)


@pytest.mark.parametrize("mesh_aggr", ["sum", "mean"])
@pytest.mark.parametrize("output_std", [False, True])
def test_graph_lam_processor_layers_zero_forward_and_backward(
    dummy_datastore_with_graph, mesh_aggr, output_std
):
    """GraphLAM should support forward and backward passes and state_dict
    roundtripping with processor_layers=0."""
    predictor = GraphLAM(
        datastore=dummy_datastore_with_graph,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=0,
        mesh_aggr=mesh_aggr,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=output_std,
    )
    forecaster = DeterministicARForecaster(
        predictor, dummy_datastore_with_graph
    )

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = dummy_datastore_with_graph.get_num_data_vars(category="state")
    d_forcing = (
        dummy_datastore_with_graph.get_num_data_vars(category="forcing") * 3
    )
    init_states = torch.randn(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.randn(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.randn(B, 1, num_grid_nodes, d_state)

    prediction, pred_std = forecaster(
        init_states, forcing_features, boundary_states
    )

    assert prediction.shape == (B, 1, num_grid_nodes, d_state)
    assert torch.isfinite(prediction).all()
    if output_std:
        assert pred_std is not None
        assert pred_std.shape == (B, 1, num_grid_nodes, d_state)
        assert torch.isfinite(pred_std).all()
        loss = prediction.sum() + pred_std.sum()
    else:
        assert pred_std is None
        loss = prediction.sum()

    # Ensure gradients propagate through zero-layer processor
    loss.backward()
    for name, param in predictor.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("m2m_embedder"):
            # m2m edge attributes are only used by processor layers
            assert param.grad is None
        else:
            assert param.grad is not None, f"Missing grad for {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite grad in {name}"
            )

    # Verify state_dict serialization and deserialization
    state_dict = predictor.state_dict()
    new_predictor = GraphLAM(
        datastore=dummy_datastore_with_graph,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=0,
        mesh_aggr=mesh_aggr,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=output_std,
    )
    new_predictor.load_state_dict(state_dict)


@pytest.mark.parametrize("invalid_layers", [-1, -5])
def test_graph_lam_negative_processor_layers_raises(
    dummy_datastore_with_graph, invalid_layers
):
    """GraphLAM should raise a ValueError when processor_layers is negative."""
    with pytest.raises(
        ValueError, match="processor_layers must be non-negative"
    ):
        GraphLAM(
            datastore=dummy_datastore_with_graph,
            graph_name="1level",
            processor_layers=invalid_layers,
        )


def test_graph_lam_positive_processor_layers_forward_prediction(
    dummy_datastore_with_graph,
):
    """Existing behavior for positive processor layer counts should remain
    unaffected."""
    predictor = GraphLAM(
        datastore=dummy_datastore_with_graph,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    assert isinstance(predictor.processor, torch.nn.Module)

    forecaster = DeterministicARForecaster(
        predictor, dummy_datastore_with_graph
    )
    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = dummy_datastore_with_graph.get_num_data_vars(category="state")
    d_forcing = (
        dummy_datastore_with_graph.get_num_data_vars(category="forcing") * 3
    )
    init_states = torch.randn(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.randn(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.randn(B, 1, num_grid_nodes, d_state)

    with torch.no_grad():
        prediction, pred_std = forecaster(
            init_states, forcing_features, boundary_states
        )

    assert prediction.shape == (B, 1, num_grid_nodes, d_state)
    assert pred_std is None
    assert torch.isfinite(prediction).all()
