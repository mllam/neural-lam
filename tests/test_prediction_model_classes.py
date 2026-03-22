# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.ar_forecaster import ARForecaster
from neural_lam.models.forecaster_module import ForecasterModule
from neural_lam.models.step_predictor import StepPredictor
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


class NoStaticDummyDatastore(DummyDatastore):
    """DummyDatastore variant that returns None for static features."""

    def get_dataarray(self, category, split, standardize=False):
        if category == "static":
            return None
        return super().get_dataarray(category, split, standardize=standardize)


class MockStepPredictor(StepPredictor):
    def __init__(self, config, datastore, **kwargs):
        super().__init__(config, datastore, **kwargs)

    def forward(self, prev_state, prev_prev_state, forcing):
        # Return zeros for state
        # The true state will be mixed in at boundaries
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.zeros_like(prev_state) if self.output_std else None
        return pred_state, pred_std


def test_ar_forecaster_unroll():
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=False,
    )

    forecaster = ARForecaster(predictor, datastore)

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


def test_forecaster_module_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
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
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
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
        config=config,
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
    load_forecaster = ARForecaster(load_predictor, datastore)

    # Load from checkpoint
    loaded_model = ForecasterModule.load_from_checkpoint(
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


def test_forecaster_module_old_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
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
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
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

    # Manually hack the checkpoint to emulate a pre-refactor state dict
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
    torch.save(ckpt, ckpt_path)

    # Build a fresh forecaster structure for loading weights into
    load_predictor = predictor_class(
        config=config,
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
    load_forecaster = ARForecaster(load_predictor, datastore)

    # Load from hacked old checkpoint
    loaded_model = ForecasterModule.load_from_checkpoint(
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


def test_step_predictor_no_static_features():
    """Model should run correctly when the datastore has no static features,
    using an empty (N, 0) tensor in place of static features."""
    datastore = NoStaticDummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=False,
    )

    # Static features buffer should exist but be empty (zero width)
    assert predictor.grid_static_features.shape == (
        datastore.num_grid_points,
        0,
    )

    # Verify a forward pass works end-to-end via ARForecaster
    forecaster = ARForecaster(predictor, datastore)
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


def test_fold_unfold_equivalence():
    """Folding (S, B, ...) into (S*B, ...) before ARForecaster and keeping
    prediction folded must match running each sample independently.

    This confirms ARForecaster's internal indexing is rank-transparent:
    init_states[:, 0/1] selects along the conditioning-step axis (dim 1),
    not the batch axis, so a folded leading dim is invisible to the rollout.
    The test also confirms that no unfold is needed inside forecast_for_batch
    for the existing logging/aggregation paths to remain correct.
    """
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(config=config, datastore=datastore)
    forecaster = ARForecaster(predictor, datastore)

    S, B = 3, 2
    N = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")
    pred_steps = 4

    torch.manual_seed(42)
    init_states = torch.randn(S, B, 2, N, d_state)
    forcing = torch.randn(S, B, pred_steps, N, d_forcing)
    boundary = torch.randn(S, B, pred_steps, N, d_state)

    # Run with (S, B) folded into one effective batch dim
    with torch.no_grad():
        pred_folded, _ = forecaster(
            init_states.flatten(0, 1),
            forcing.flatten(0, 1),
            boundary.flatten(0, 1),
        )
    pred_folded = pred_folded.unflatten(0, (S, B))

    # Run each sample independently and stack
    with torch.no_grad():
        pred_explicit = torch.stack(
            [forecaster(init_states[s], forcing[s], boundary[s])[0]
             for s in range(S)]
        )

    assert torch.allclose(pred_folded, pred_explicit)
