"""Tests for the PersistencePredictor baseline model."""

# Third-party
import pytest
import torch

# First-party
from neural_lam.models import ARForecaster
from neural_lam.models.step_predictors.persistence import PersistencePredictor
from tests.conftest import init_datastore_example


def test_persistence_predictor_returns_prev_state():
    """PersistencePredictor.forward returns prev_state unchanged."""
    datastore = init_datastore_example("mdp")
    predictor = PersistencePredictor(datastore=datastore)
    assert predictor.trainable is False

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    prev_state = torch.randn(B, num_grid_nodes, d_state)
    prev_prev_state = torch.randn(B, num_grid_nodes, d_state)
    forcing = torch.randn(B, num_grid_nodes, d_forcing)

    pred_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert pred_std is None
    assert torch.equal(pred_state, prev_state)


def test_persistence_predictor_ignores_kwargs():
    """Extra graph-specific kwargs are silently absorbed."""
    datastore = init_datastore_example("mdp")
    predictor = PersistencePredictor(
        datastore=datastore,
        graph_name="multiscale",
        hidden_dim=64,
        hidden_layers=1,
        processor_layers=4,
        mesh_aggr="sum",
    )
    assert isinstance(predictor, PersistencePredictor)


def test_persistence_forecaster_unroll():
    """ARForecaster with PersistencePredictor reproduces initial state."""
    datastore = init_datastore_example("mdp")
    predictor = PersistencePredictor(datastore=datastore)
    forecaster = ARForecaster(predictor, datastore)

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * 3
    pred_steps = 4

    init_states = torch.randn(B, 2, num_grid_nodes, d_state)
    forcing = torch.randn(B, pred_steps, num_grid_nodes, d_forcing)
    boundary_states = torch.randn(B, pred_steps, num_grid_nodes, d_state)

    prediction, pred_std = forecaster(init_states, forcing, boundary_states)

    assert prediction.shape == (B, pred_steps, num_grid_nodes, d_state)
    assert pred_std is None

    # Persistence keeps every interior node at init_states[:, 1] for all
    # steps, while boundary nodes are overwritten with boundary_states.
    interior_mask = forecaster.interior_mask.squeeze(0).squeeze(-1).bool()
    boundary_mask = forecaster.boundary_mask.squeeze(0).squeeze(-1).bool()
    for t in range(pred_steps):
        assert torch.equal(
            prediction[:, t, interior_mask, :],
            init_states[:, 1, interior_mask, :],
        )
        assert torch.equal(
            prediction[:, t, boundary_mask, :],
            boundary_states[:, t, boundary_mask, :],
        )


def test_persistence_predicts_std_false():
    """
    PersistencePredictor.predicts_std is always False and logs warning
    when output_std is True.
    """
    # Standard library
    from unittest.mock import patch

    datastore = init_datastore_example("mdp")
    target_patch = (
        "neural_lam.models.step_predictors.persistence.logger.warning"
    )
    with patch(target_patch) as mock_warn:
        predictor = PersistencePredictor(datastore=datastore, output_std=True)
        assert not predictor.predicts_std
        mock_warn.assert_called_once_with(
            "Persistence predictor does not support predicting "
            "standard deviation. The output_std parameter will be ignored."
        )


def test_persistence_training_error():
    """ValueError must be raised if trying to train the persistence model."""
    # Standard library
    from unittest.mock import MagicMock, patch

    # First-party
    from neural_lam.train_model import main

    mock_args = MagicMock()
    mock_args.eval = None  # training mode
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.model = "persistence"
    mock_args.devices = ["auto"]

    mock_predictor = MagicMock()
    mock_predictor.trainable = False

    with (
        patch(
            "neural_lam.train_model.ArgumentParser.parse_args",
            return_value=mock_args,
        ),
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch(
            "neural_lam.train_model.MODELS",
            {"persistence": MagicMock(return_value=mock_predictor)},
        ),
        pytest.raises(
            ValueError,
            match="The persistence model cannot be trained",
        ),
    ):
        getattr(main, "__wrapped__", main)()
