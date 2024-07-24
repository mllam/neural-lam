# Third-party
import pytest
from test_datastores import DATASTORES, init_datastore

# First-party
from neural_lam.weather_dataset import WeatherDataset


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item(datastore_name):
    """Check that the `datastore.get_dataarray` method is implemented.

    Validate the shapes of the tensors match between the different
    components of the training sample.

    init_states: (2, N_grid, d_features)
    target_states: (ar_steps, N_grid, d_features)
    forcing: (ar_steps, N_grid, d_windowed_forcing) # batch_times: (ar_steps,)
    """
    datastore = init_datastore(datastore_name)
    N_gridpoints = datastore.grid_shape_state.x * datastore.grid_shape_state.y

    N_pred_steps = 4
    forcing_window_size = 3
    dataset = WeatherDataset(
        datastore=datastore,
        batch_size=1,
        split="train",
        ar_steps=N_pred_steps,
        forcing_window_size=forcing_window_size,
    )

    item = dataset[0]

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    init_states, target_states, forcing, batch_times = item

    # initial states
    assert init_states.shape[0] == 2  # two time steps go into the input
    assert init_states.shape[1] == N_gridpoints
    assert init_states.shape[2] == datastore.get_num_data_vars("state")

    # output states
    assert target_states.shape[0] == N_pred_steps
    assert target_states.shape[1] == N_gridpoints
    assert target_states.shape[2] == datastore.get_num_data_vars("state")

    # forcing
    assert forcing.shape[0] == N_pred_steps  # number of prediction steps
    assert forcing.shape[1] == N_gridpoints  # number of grid points
    # number of features x window size
    assert (
        forcing.shape[2]
        == datastore.get_num_data_vars("forcing") * forcing_window_size
    )

    # batch times
    assert batch_times.shape[0] == N_pred_steps
