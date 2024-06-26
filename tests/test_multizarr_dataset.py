# Standard library
import os

# First-party
from create_mesh import main as create_mesh
from neural_lam.datastore.multizarr import MultiZarrDatastore
# from neural_lam.datasets.config import Config
from neural_lam.weather_dataset import WeatherDataset

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_DISABLED"] = "true"


def test_load_analysis_dataset():
    # TODO: Access rights should be fixed for pooch to work
    datastore = MultiZarrDatastore(
        config_path="tests/datastore_configs/multizarr.danra.yaml"
    )

    var_state_names = datastore.get_vars_names(category="state")
    var_state_units = datastore.get_vars_units(category="state")
    num_state_vars = datastore.get_num_data_vars(category="state")

    assert len(var_state_names) == len(var_state_units) == num_state_vars

    var_forcing_names = datastore.get_vars_names(category="forcing")
    var_forcing_units = datastore.get_vars_units(category="forcing")
    num_forcing_vars = datastore.get_num_data_vars(category="forcing")

    assert len(var_forcing_names) == len(var_forcing_units) == num_forcing_vars
    
    stats = datastore.get_normalization_stats(category="state")


    # Assert dataset can be loaded
    ds = datastore.get_dataarray(category="state")
    grid = ds.sizes["y"] * ds.sizes["x"]
    dataset = WeatherDataset(datastore=datastore, split="train", ar_steps=3, standardize=True)
    batch = dataset[0]
    # return init_states, target_states, forcing, batch_times
    # init_states: (2, N_grid, d_features)
    # target_states: (ar_steps-2, N_grid, d_features)
    # forcing: (ar_steps-2, N_grid, d_windowed_forcing)
    # batch_times: (ar_steps-2,)
    assert list(batch[0].shape) == [2, grid, num_state_vars]
    assert list(batch[1].shape) == [dataset.ar_steps - 2, grid, num_state_vars]
    assert list(batch[2].shape) == [
        dataset.ar_steps - 2,
        grid,
        num_forcing_vars * config.forcing.window,
    ]
    assert isinstance(batch[3], list)

    # Assert provided grid-shapes
    assert config.get_xy("static")[0].shape == (
        config.grid_shape_state.y,
        config.grid_shape_state.x,
    )
    assert config.get_xy("static")[0].shape == (ds.sizes["y"], ds.sizes["x"])


def test_create_graph_analysis_dataset():
    args = [
        "--graph=hierarchical",
        "--hierarchical=1",
        "--data_config=tests/data_config.yaml",
        "--levels=2",
    ]
    create_mesh(args)


# def test_train_model_analysis_dataset():
#     args = [
#         "--model=hi_lam",
#         "--data_config=tests/data_config.yaml",
#         "--num_workers=4",
#         "--epochs=1",
#         "--graph=hierarchical",
#         "--hidden_dim=16",
#         "--hidden_layers=1",
#         "--processor_layers=1",
#         "--ar_steps_eval=1",
#         "--eval=val",
#         "--n_example_pred=0",
#         "--val_steps_to_log=1",
#     ]
#     train_model(args)
