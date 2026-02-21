# Standard library
import os
from argparse import Namespace

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

ModelArgs = Namespace(
    output_std=False,
    num_past_forcing_steps=1,
    num_future_forcing_steps=1,
    loss="mse",
    lr=1e-3,
    restore_opt=False,
    n_example_pred=1,
    graph="1level",
    hidden_dim=4,
    hidden_layers=1,
    processor_layers=1,
    mesh_aggr="sum",
    val_steps_to_log=[1],
    metrics_watch=[],
)


class MockStepPredictor(StepPredictor):
    def __init__(self, args, config, datastore):
        super().__init__(args, config, datastore)

    def forward(self, prev_state, prev_prev_state, forcing):
        # Return zeros for state
        # The true state will be mixed in at boundaries
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.zeros_like(prev_state) if self.output_std else None
        return pred_state, pred_std


def test_ar_forecaster_unroll():
    datastore = init_datastore_example("mdp")
    MinimalArgs = Namespace(
        output_std=False,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(MinimalArgs, config, datastore)

    # Mocking explicit interior and boundary to test masking
    predictor.interior_mask = torch.zeros_like(predictor.interior_mask)
    predictor.interior_mask[0, 0] = 1  # One node is interior
    predictor.boundary_mask = 1 - predictor.interior_mask

    forecaster = ARForecaster(predictor)

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        MinimalArgs.num_past_forcing_steps
        + MinimalArgs.num_future_forcing_steps
        + 1
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

    model = ForecasterModule("graph_lam", ModelArgs, config, datastore)

    ckpt_path = tmp_path / "test.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    # Load from checkpoint
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([Namespace]):
            loaded_model = ForecasterModule.load_from_checkpoint(
                ckpt_path, datastore=datastore
            )
    else:
        loaded_model = ForecasterModule.load_from_checkpoint(
            ckpt_path, datastore=datastore
        )

    # Validate the correct internal hierarchy has been constructed
    assert loaded_model.forecaster.predictor.__class__.__name__ == "GraphLAM"

    # Verify that outputs match (checkpoint successfully restored weights)
    B, num_grid_nodes = 2, model.forecaster.predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        ModelArgs.num_past_forcing_steps
        + ModelArgs.num_future_forcing_steps
        + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    border_states = torch.ones(B, 1, num_grid_nodes, d_state) * 5.0

    with torch.no_grad():
        out_before = model.forecaster(
            init_states, forcing_features, border_states
        )
        out_after = loaded_model.forecaster(
            init_states, forcing_features, border_states
        )

    assert torch.allclose(out_before[0], out_after[0])
