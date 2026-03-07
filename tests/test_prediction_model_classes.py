# Standard library
import os

# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.ar_forecaster import ARForecaster
from neural_lam.models.forecaster_module import ForecasterModule
from neural_lam.models.step_predictor import StepPredictor
from tests.conftest import init_datastore_example

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


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
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )

    forecaster = ARForecaster(predictor, datastore)

    # Mocking explicit interior and boundary to test masking (Item 7:
    # masks now live on ARForecaster, not StepPredictor)
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

    # Build forecaster externally (Items 2 & 3)
    from neural_lam.models import MODELS
    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph="1level",
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
        output_std=False,
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
        graph="1level",
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
        ckpt_path, datastore=datastore, forecaster=load_forecaster
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

    from neural_lam.models import MODELS
    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph="1level",
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
        output_std=False,
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
        graph="1level",
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
        ckpt_path, datastore=datastore, forecaster=load_forecaster
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
