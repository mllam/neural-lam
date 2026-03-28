# Third-party
import torch
import pytest

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.ar_model import ARModel
from tests.dummy_datastore import DummyDatastore


class DummyArgs:
    output_std = False
    loss = "mse"
    restore_opt = False
    n_example_pred = 0
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    lr = 1.0e-3
    val_steps_to_log = [1]
    metrics_watch = []
    var_leads_metrics_watch = {}


class DummyARModel(ARModel):
    def predict_step(self, prev_state, prev_prev_state, forcing):
        del prev_prev_state, forcing
        return prev_state, None


def test_ar_model_initializes_core_training_state():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=""
        )
    )

    model = DummyARModel(args=DummyArgs(), config=config, datastore=datastore)

    assert model.grid_static_features.shape == (16, 1)
    assert model.num_grid_nodes == 16
    assert model.grid_dim == 17
    assert model.grid_output_dim == datastore.get_num_data_vars("state")
    assert model.boundary_mask.shape == (16, 1)
    assert model.interior_mask.shape == (16, 1)
    assert model.feature_weights.shape == (datastore.get_num_data_vars("state"),)


def test_ar_model_forward_samples_ensemble_when_enabled():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=""
        ),
        training=nlconfig.TrainingConfig(
            output_mode="ensemble",
            ensemble_size=3,
        ),
    )
    model = DummyARModel(args=DummyArgs(), config=config, datastore=datastore)

    preds = model(torch.ones(2, 4, 5))

    assert preds.shape == (3, 2, 4, 5)


def test_all_gather_cat_returns_input_without_trainer():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=""
        )
    )
    model = DummyARModel(args=DummyArgs(), config=config, datastore=datastore)
    tensor = torch.randn(2, 3, 4)

    gathered = model.all_gather_cat(tensor)

    assert torch.equal(gathered, tensor)


def test_ar_model_raises_error_on_incompatible_loss_and_output_std():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=""
        )
    )

    class IncompatibleArgs(DummyArgs):
        output_std = True
        loss = "wmse"

    with pytest.raises(
        ValueError, match="is incompatible with loss function 'wmse'"
    ):
        DummyARModel(
            args=IncompatibleArgs(), config=config, datastore=datastore
        )

