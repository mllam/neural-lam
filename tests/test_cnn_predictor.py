# Third-party
import pytest
import torch

# First-party
from neural_lam.models import ARForecaster, CNNPredictor
from tests.dummy_datastore import DummyDatastore


class NoStaticDummyDatastore(DummyDatastore):
    """Dummy datastore variant without static features."""

    def get_dataarray(self, category, split, standardize=False):
        if category == "static":
            return None
        return super().get_dataarray(category, split, standardize=standardize)


def _make_predictor(datastore, output_std=False, cnn_film=False):
    return CNNPredictor(
        datastore=datastore,
        cnn_channels=8,
        cnn_blocks=2,
        cnn_se_reduction=4,
        cnn_film=cnn_film,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=output_std,
    )


def _make_inputs(datastore, batch_size=2, pred_steps=1):
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * 3

    prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    prev_prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    forcing = torch.randn(batch_size, num_grid_nodes, d_forcing)

    init_states = torch.stack((prev_prev_state, prev_state), dim=1)
    forcing_features = torch.randn(
        batch_size,
        pred_steps,
        num_grid_nodes,
        d_forcing,
    )
    boundary_states = torch.randn(
        batch_size,
        pred_steps,
        num_grid_nodes,
        d_state,
    )

    return (
        prev_state,
        prev_prev_state,
        forcing,
        init_states,
        forcing_features,
        boundary_states,
    )


def test_cnn_predictor_forward_shape_without_std():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore, output_std=False)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    prediction, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert prediction.shape == prev_state.shape
    assert pred_std is None


def test_cnn_predictor_forward_shape_with_std():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore, output_std=True)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    prediction, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert prediction.shape == prev_state.shape
    assert pred_std.shape == prev_state.shape
    assert torch.all(pred_std > 0)


def test_cnn_predictor_forward_shape_with_film():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore, cnn_film=True)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    prediction, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert prediction.shape == prev_state.shape
    assert pred_std is None


def test_cnn_predictor_no_static_features():
    datastore = NoStaticDummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    prediction, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert predictor.grid_static_features.shape == (
        datastore.num_grid_points,
        0,
    )
    assert prediction.shape == prev_state.shape
    assert pred_std is None


def test_cnn_predictor_ar_forecaster_rollout_shape():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore)
    forecaster = ARForecaster(predictor, datastore)
    _, _, _, init_states, forcing_features, boundary_states = _make_inputs(
        datastore,
        pred_steps=3,
    )

    prediction, pred_std = forecaster(
        init_states,
        forcing_features,
        boundary_states,
    )

    assert prediction.shape == boundary_states.shape
    assert pred_std is None


def test_cnn_predictor_rejects_non_regular_datastore():
    with pytest.raises(TypeError, match="BaseRegularGridDatastore"):
        CNNPredictor(datastore=object())


def test_cnn_predictor_rejects_wrong_forcing_size():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    with pytest.raises(ValueError, match="forcing feature dimension"):
        predictor(prev_state, prev_prev_state, forcing[..., :-1])


def test_cnn_predictor_rejects_wrong_state_size():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = _make_predictor(datastore)
    prev_state, prev_prev_state, forcing, *_ = _make_inputs(datastore)

    with pytest.raises(ValueError, match="state feature dimension"):
        predictor(prev_state[..., :-1], prev_prev_state[..., :-1], forcing)
