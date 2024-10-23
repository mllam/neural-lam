# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item_shapes(datastore_name):
    """Check that the `datastore.get_dataarray` method is implemented.

    Validate the shapes of the tensors match between the different
    components of the training sample.

    init_states: (2, N_grid, d_features)
    target_states: (ar_steps, N_grid, d_features)
    forcing: (ar_steps, N_grid, d_windowed_forcing) # batch_times: (ar_steps,)

    """
    datastore = init_datastore_example(datastore_name)
    N_gridpoints = datastore.num_grid_points

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
    init_states, target_states, forcing, target_times = item

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
        forcing.shape[2]
        == datastore.get_num_data_vars("forcing") * forcing_window_size
    )

    # batch times
    assert target_times.ndim == 1
    assert target_times.shape[0] == N_pred_steps

    # try to get the last item of the dataset to ensure slicing and stacking
    # operations are working as expected and are consistent with the dataset
    # length
    dataset[len(dataset) - 1]


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item_create_dataarray_from_tensor(datastore_name):
    datastore = init_datastore_example(datastore_name)

    N_pred_steps = 4
    forcing_window_size = 3
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=N_pred_steps,
        forcing_window_size=forcing_window_size,
    )

    idx = 0

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    _, target_states, _, target_times_arr = dataset[idx]
    _, da_target_true, _, da_target_times_true = dataset._build_item_dataarrays(
        idx=idx
    )

    target_times = np.array(target_times_arr, dtype="datetime64[ns]")
    np.testing.assert_equal(target_times, da_target_times_true.values)

    da_target = dataset.create_dataarray_from_tensor(
        tensor=target_states, category="state", time=target_times
    )

    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target.values, da_target_true.values, rtol=1e-6
    )
    assert da_target.dims == da_target_true.dims
    for dim in da_target.dims:
        np.testing.assert_equal(
            da_target[dim].values, da_target_true[dim].values
        )

    if isinstance(datastore, BaseRegularGridDatastore):
        # test unstacking the grid coordinates
        da_target_unstacked = datastore.unstack_grid_coords(da_target)
        assert all(
            coord_name in da_target_unstacked.coords
            for coord_name in ["x", "y"]
        )

    # check construction of a single time
    da_target_single = dataset.create_dataarray_from_tensor(
        tensor=target_states[0], category="state", time=target_times[0]
    )

    # check that the content is the same
    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target_single.values, da_target_true[0].values, rtol=1e-6
    )
    assert da_target_single.dims == da_target_true[0].dims
    for dim in da_target_single.dims:
        np.testing.assert_equal(
            da_target_single[dim].values, da_target_true[0][dim].values
        )

    if isinstance(datastore, BaseRegularGridDatastore):
        # test unstacking the grid coordinates
        da_target_single_unstacked = datastore.unstack_grid_coords(
            da_target_single
        )
        assert all(
            coord_name in da_target_single_unstacked.coords
            for coord_name in ["x", "y"]
        )


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_single_batch(datastore_name, split):
    """Check that the `datasto re.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.

    """
    datastore = init_datastore_example(datastore_name)

    device_name = (
        torch.device("cuda") if torch.cuda.is_available() else "cpu"
    )  # noqa

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
        forcing_window_size = 3

    args = ModelArgs()

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    def _create_graph():
        if not graph_dir_path.exists():
            create_graph_from_datastore(
                datastore=datastore,
                output_root_path=str(graph_dir_path),
                n_max_levels=1,
            )

    if not isinstance(datastore, BaseRegularGridDatastore):
        with pytest.raises(NotImplementedError):
            _create_graph()
        pytest.skip("Skipping on model-run on non-regular grid datastores")

    dataset = WeatherDataset(datastore=datastore, split=split)

    model = GraphLAM(  # noqa
        args=args,
        datastore=datastore,
    )

    model_device = model.to(device_name)
    data_loader = DataLoader(dataset, batch_size=5)
    batch = next(iter(data_loader))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)
    model_device.training_step(batch_device)
