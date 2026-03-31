# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graph_lam import GraphLAM
from tests.conftest import init_datastore_example


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
