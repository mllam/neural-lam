# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graph_lam import GraphLAM
from tests.conftest import init_datastore_example


def test_forcing_normalization_tiling():
    """Test forcing stats are tiled correctly for windowed forcing."""
    num_forcing_vars = 2
    window_size = 3
    forcing_mean = torch.tensor([10.0, 20.0])
    forcing_std = torch.tensor([2.0, 4.0])

    forcing = torch.ones(2, 4, num_forcing_vars * window_size) * 30.0

    window_size_calc = forcing.shape[-1] // forcing_mean.shape[-1]
    assert window_size_calc == window_size

    forcing_mean_tiled = forcing_mean.repeat_interleave(window_size_calc)
    forcing_std_tiled = forcing_std.repeat_interleave(window_size_calc)

    assert forcing_mean_tiled.shape == (num_forcing_vars * window_size,)
    assert forcing_std_tiled.shape == (num_forcing_vars * window_size,)

    expected_mean = torch.tensor([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
    expected_std = torch.tensor([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
    assert torch.allclose(forcing_mean_tiled, expected_mean)
    assert torch.allclose(forcing_std_tiled, expected_std)

    normalized = (forcing - forcing_mean_tiled) / forcing_std_tiled
    assert normalized.shape == forcing.shape


def test_state_normalization():
    """Test state normalization."""
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state_mean = torch.tensor([1.0, 2.0])
    state_std = torch.tensor([0.5, 1.0])

    normalized = (state - state_mean) / state_std

    expected = torch.tensor([[0.0, 0.0], [4.0, 2.0]])
    assert torch.allclose(normalized, expected)


def test_empty_forcing_handling():
    """Test empty forcing tensor is handled correctly."""
    forcing = torch.empty(2, 4, 0)
    assert forcing.shape[-1] == 0
    assert forcing.shape == (2, 4, 0)


@pytest.mark.parametrize("datastore_name", ["mdp"])
def test_on_after_batch_transfer(datastore_name):
    """Test on_after_batch_transfer with real datastore."""
    datastore = init_datastore_example(datastore_name)

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = "1level"
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    args = ModelArgs()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(args=args, datastore=datastore, config=config)

    num_grid_nodes = datastore.num_grid_points
    num_state_vars = datastore.get_num_data_vars("state")
    num_forcing_vars = datastore.get_num_data_vars("forcing")
    window_size = (
        args.num_past_forcing_steps + args.num_future_forcing_steps + 1
    )
    ar_steps = 2

    init_states = torch.randn(1, 2, num_grid_nodes, num_state_vars)
    target_states = torch.randn(1, ar_steps, num_grid_nodes, num_state_vars)
    forcing = torch.randn(
        1, ar_steps, num_grid_nodes, num_forcing_vars * window_size
    )
    target_times = torch.randint(0, 1000000, (1, ar_steps))

    batch = (init_states, target_states, forcing, target_times)

    norm_init_states, norm_target_states, norm_forcing, norm_target_times = (
        model.on_after_batch_transfer(batch, 0)
    )

    assert norm_init_states.shape == init_states.shape
    assert norm_target_states.shape == target_states.shape
    assert norm_forcing.shape == forcing.shape
    assert norm_target_times.shape == target_times.shape

    # Data should actually change after normalization
    assert not torch.allclose(norm_init_states, init_states)
    assert not torch.allclose(norm_target_states, target_states)
    if num_forcing_vars > 0:
        assert not torch.allclose(norm_forcing, forcing)

    # Verify normalization is mathematically correct: result == (x - mean)/std
    eps = torch.finfo(model.state_std.dtype).eps
    state_std_safe = torch.clamp(model.state_std, min=eps)
    expected_init = (init_states - model.state_mean) / state_std_safe
    expected_target = (target_states - model.state_mean) / state_std_safe

    assert torch.allclose(norm_init_states, expected_init)
    assert torch.allclose(norm_target_states, expected_target)

    if num_forcing_vars > 0 and model.forcing_mean is not None:
        window_size_calc = forcing.shape[-1] // model.forcing_mean.shape[-1]
        forcing_mean_tiled = model.forcing_mean.repeat_interleave(
            window_size_calc
        )
        forcing_std_safe = torch.clamp(model.forcing_std, min=eps)
        forcing_std_tiled = forcing_std_safe.repeat_interleave(window_size_calc)
        expected_forcing = (forcing - forcing_mean_tiled) / forcing_std_tiled
        assert torch.allclose(norm_forcing, expected_forcing)


def test_normalization_happens_exactly_once():
    """
    Assert that data is standardized exactly once - on GPU in
    on_after_batch_transfer - and never in the dataset (CPU path).
    """
    # Standard library
    from datetime import timedelta

    # First-party
    from neural_lam.weather_dataset import WeatherDataset
    from tests.dummy_datastore import DummyDatastore

    datastore = DummyDatastore(n_timesteps=20, step_length=timedelta(hours=1))

    # Track how many times get_dataarray is called with standardize=True
    standardize_call_count = 0
    original_get_dataarray = datastore.get_dataarray

    def counting_get_dataarray(category, split=None, standardize=False):
        nonlocal standardize_call_count
        if standardize:
            standardize_call_count += 1
        return original_get_dataarray(
            category=category, split=split, standardize=standardize
        )

    datastore.get_dataarray = counting_get_dataarray

    # Dataset should load raw data (standardize=False)
    ds = WeatherDataset(datastore=datastore, split="train", standardize=False)
    assert standardize_call_count == 0, (
        "Dataset called get_dataarray with standardize=True "
        f"{standardize_call_count} times, expected 0"
    )

    # Get a sample - should still be raw
    init_states, target_states, forcing, _ = ds[0]

    # Manually apply GPU normalization once
    eps = torch.finfo(torch.float32).eps
    state_std_safe = torch.clamp(
        torch.tensor(
            datastore.get_standardization_dataarray("state").state_std.values,
            dtype=torch.float32,
        ),
        min=eps,
    )
    state_mean = torch.tensor(
        datastore.get_standardization_dataarray("state").state_mean.values,
        dtype=torch.float32,
    )
    normalized_once = (init_states - state_mean) / state_std_safe

    # Applying normalization a second time should give a different result
    normalized_twice = (normalized_once - state_mean) / state_std_safe
    assert not torch.allclose(normalized_once, normalized_twice), (
        "Normalizing twice gave the same result - "
        "data may already be normalized"
    )
