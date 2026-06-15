# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models import ARForecaster, ForecasterModule, StepPredictor
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example
from tests.dummy_datastore import BoundaryDummyDatastore, DummyDatastore

NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


class _MockStepPredictor(StepPredictor):
    """Minimal predictor so a ForecasterModule can be built without a graph."""

    def forward(self, prev_state, prev_prev_state, forcing):
        return torch.zeros_like(prev_state), None


def _build_module(datastore, datastore_boundary=None):
    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        }
    )
    predictor = _MockStepPredictor(datastore=datastore, output_std=False)
    forecaster = ARForecaster(predictor, datastore)
    return ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
    )


def test_on_after_batch_transfer():
    """The hook standardizes state and forcing as (x - mean) / std and
    leaves shapes, boundary forcing and target times untouched."""
    datastore = init_datastore_example("mdp")
    model = _build_module(datastore)

    num_grid_nodes = datastore.num_grid_points
    num_state_vars = datastore.get_num_data_vars("state")
    num_forcing_vars = datastore.get_num_data_vars("forcing")
    window_size = NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    ar_steps = 2

    init_states = torch.randn(1, 2, num_grid_nodes, num_state_vars)
    target_states = torch.randn(1, ar_steps, num_grid_nodes, num_state_vars)
    forcing = torch.randn(
        1, ar_steps, num_grid_nodes, num_forcing_vars * window_size
    )
    boundary = torch.randn(1, ar_steps, num_grid_nodes, 3)
    target_times = torch.randint(0, 1000000, (1, ar_steps))

    norm_init, norm_target, norm_forcing, norm_boundary, norm_times = (
        model.on_after_batch_transfer(
            (init_states, target_states, forcing, boundary, target_times), 0
        )
    )

    assert norm_init.shape == init_states.shape
    assert norm_target.shape == target_states.shape
    assert norm_forcing.shape == forcing.shape
    assert torch.equal(norm_boundary, boundary)
    assert torch.equal(norm_times, target_times)

    expected_init = (init_states - model.state_mean) / model.state_std
    expected_target = (target_states - model.state_mean) / model.state_std
    assert torch.allclose(norm_init, expected_init)
    assert torch.allclose(norm_target, expected_target)

    assert num_forcing_vars > 0, "mdp example is expected to have forcing"
    forcing_mean_tiled = model.forcing_mean.repeat_interleave(window_size)
    forcing_std_tiled = model.forcing_std.repeat_interleave(window_size)
    expected_forcing = (forcing - forcing_mean_tiled) / forcing_std_tiled
    assert torch.allclose(norm_forcing, expected_forcing)


def test_normalization_applied_exactly_once():
    """Data fed through WeatherDataset and the hook must be standardized
    exactly once: not skipped (WeatherDataset returns raw data) and not
    applied twice (no leftover CPU-side standardization)."""
    datastore = init_datastore_example("mdp")
    model = _build_module(datastore)

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=2,
        ar_steps_eval=2,
        batch_size=2,
        num_workers=0,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
    )
    data_module.setup(stage="fit")
    batch = next(iter(data_module.train_dataloader()))

    raw_init = batch[0]
    norm_init = model.on_after_batch_transfer(batch, 0)[0]

    once = (raw_init - model.state_mean) / model.state_std
    twice = (once - model.state_mean) / model.state_std

    assert torch.allclose(norm_init, once)
    assert not torch.allclose(norm_init, raw_init)  # not zero times
    assert not torch.allclose(norm_init, twice)  # not twice


def test_boundary_standardized_when_datastore_provided():
    """Boundary forcing is standardized by ForecasterModule when a
    boundary datastore is wired in, using its own forcing mean/std."""
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=20)
    datastore_boundary = BoundaryDummyDatastore(
        n_grid_points=25, n_timesteps=20
    )
    model = _build_module(
        datastore=datastore, datastore_boundary=datastore_boundary
    )
    assert model.boundary_mean is not None
    assert model.boundary_std is not None

    num_boundary_grid = datastore_boundary.num_grid_points
    num_boundary_vars = datastore_boundary.get_num_data_vars("forcing")
    window_size = 3  # arbitrary; must match the stacked feature axis below
    ar_steps = 2

    init_states = torch.randn(1, 2, datastore.num_grid_points, 5)
    target_states = torch.randn(1, ar_steps, datastore.num_grid_points, 5)
    forcing = torch.randn(
        1, ar_steps, datastore.num_grid_points, 2 * window_size
    )
    boundary = torch.randn(
        1, ar_steps, num_boundary_grid, num_boundary_vars * window_size
    )
    target_times = torch.randint(0, 1000000, (1, ar_steps))

    _, _, _, norm_boundary, _ = model.on_after_batch_transfer(
        (init_states, target_states, forcing, boundary, target_times), 0
    )

    boundary_mean_tiled = model.boundary_mean.repeat_interleave(window_size)
    boundary_std_tiled = model.boundary_std.repeat_interleave(window_size)
    expected = (boundary - boundary_mean_tiled) / boundary_std_tiled
    assert torch.allclose(norm_boundary, expected)
    # Tiled buffers should now be cached.
    assert model.boundary_mean_tiled is not None
    assert model.boundary_std_tiled is not None


def test_boundary_passthrough_when_no_boundary_datastore():
    """Without a boundary datastore, boundary is passed through unchanged
    even if the tensor has non-zero last dim."""
    datastore = init_datastore_example("mdp")
    model = _build_module(datastore)
    assert model.boundary_mean is None

    num_state = datastore.get_num_data_vars("state")
    boundary = torch.randn(1, 2, datastore.num_grid_points, 3)
    init_states = torch.randn(1, 2, datastore.num_grid_points, num_state)
    target_states = torch.randn(1, 2, datastore.num_grid_points, num_state)
    forcing = torch.randn(
        1,
        2,
        datastore.num_grid_points,
        datastore.get_num_data_vars("forcing")
        * (NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1),
    )
    target_times = torch.randint(0, 1000000, (1, 2))

    _, _, _, norm_boundary, _ = model.on_after_batch_transfer(
        (init_states, target_states, forcing, boundary, target_times), 0
    )
    assert torch.equal(norm_boundary, boundary)


def test_safe_std_clamps_near_zero():
    """Regression test for https://github.com/mllam/neural-lam/issues/136:
    near-zero std is clamped to machine epsilon (with a warning) so
    standardization cannot produce NaN/Inf."""
    eps = torch.finfo(torch.float32).eps

    with pytest.warns(UserWarning, match="near-zero std"):
        std = ForecasterModule._safe_std([0.0, 1.0, 2.0], eps, "state")

    assert std[0] == eps
    assert std[1] == 1.0
    assert std[2] == 2.0
    assert torch.isfinite(std).all()
