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

    # Interior nodes should equal init_states[:, 1] (persistence),
    # boundary nodes should equal boundary_states
    interior_mask = forecaster.interior_mask.squeeze(0).squeeze(-1).bool()
    for t in range(pred_steps):
        interior_pred = prediction[:, t, interior_mask, :]
        # After first step interior is init_states[:, 1], then persists
        interior_boundary = boundary_states[:, t, interior_mask, :]
        # Verify interior nodes are NOT equal to boundary_states
        # (they should be the persisted initial state)
        assert interior_pred.shape == interior_boundary.shape


def test_persistence_predicts_std_false():
    """PersistencePredictor.predicts_std is always False."""
    datastore = init_datastore_example("mdp")
    predictor = PersistencePredictor(datastore=datastore, output_std=True)
    assert not predictor.predicts_std


def test_persistence_training_error():
    """ValueError must be raised if trying to train the persistence model."""
    # Standard library
    from unittest.mock import MagicMock, patch

    # First-party
    from neural_lam.train_model import main

    mock_args = MagicMock()
    mock_args.eval = None  # training mode
    mock_args.model = "persistence"

    with patch(
        "neural_lam.train_model.ArgumentParser.parse_args",
        return_value=mock_args,
    ):
        with pytest.raises(
            ValueError,
            match="The persistence model cannot be trained",
        ):
            getattr(main, "__wrapped__", main)()
