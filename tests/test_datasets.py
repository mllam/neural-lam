# Standard library
from pathlib import Path

# Third-party
import pytest
import torch
from test_datastores import DATASTORES, init_datastore
from torch.utils.data import DataLoader

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item(datastore_name):
    """Check that the `datasto re.get_dataarray` method is implemented.

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
        split="train",
        ar_steps=N_pred_steps,
        forcing_window_size=forcing_window_size,
    )

    item = dataset[0]

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    init_states, target_states, forcing, batch_times = item

    # initial states
    assert init_states.ndim == 3
    assert init_states.shape[0] == 2  # two time steps go into the input
    assert init_states.shape[1] == N_gridpoints
    assert init_states.shape[2] == datastore.get_num_data_vars("state")

    # output states
    assert target_states.ndim == 3
    assert target_states.shape[0] == N_pred_steps
    assert target_states.shape[1] == N_gridpoints
    assert target_states.shape[2] == datastore.get_num_data_vars("state")

    # forcing
    assert forcing.ndim == 3
    assert forcing.shape[0] == N_pred_steps
    assert forcing.shape[1] == N_gridpoints
    assert (
        forcing.shape[2] == datastore.get_num_data_vars("forcing") * forcing_window_size
    )

    # batch times
    assert batch_times.ndim == 1
    assert batch_times.shape[0] == N_pred_steps

    # try to get the last item of the dataset to ensure slicing and stacking
    # operations are working as expected and are consistent with the dataset
    # length
    dataset[len(dataset) - 1]


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_single_batch(datastore_name, split):
    """Check that the `datasto re.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.

    """
    datastore = init_datastore(datastore_name)

    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"  # noqa

    graph_name = "1level"

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        # XXX: this should be superfluous when we have already defined the
        # model object no?
        graph = graph_name
        hidden_dim = 8
        hidden_layers = 1
        processor_layers = 4
        mesh_aggr = "sum"

    args = ModelArgs()

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    dataset = WeatherDataset(datastore=datastore, split=split)

    model = GraphLAM(  # noqa
        args=args,
        forcing_window_size=dataset.forcing_window_size,
        datastore=datastore,
    )

    model_device = model.to(device_name)
    data_loader = DataLoader(dataset, batch_size=5)
    batch = next(iter(data_loader))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)
    model_device.training_step(batch_device)
