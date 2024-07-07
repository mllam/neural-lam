# Standard library
import os

# Third-party
import pooch
import pytest

# First-party
from create_mesh import main as create_mesh
from neural_lam.weather_dataset import WeatherDataset
from neural_lam.datastore.npyfiles import NumpyFilesDatastore
from neural_lam.datastore.multizarr import MultiZarrDatastore
from train_model import main as train_model

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_DISABLED"] = "true"

# Initializing variables for the s3 client
S3_BUCKET_NAME = "mllam-testdata"
S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.1.0.zip"
S3_FULL_PATH = "/".join([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_FILE_PATH])
TEST_DATA_KNOWN_HASH = (
    "98c7a2f442922de40c6891fe3e5d190346889d6e0e97550170a82a7ce58a72b7"
)


@pytest.fixture(scope="session")
def ewc_testdata_path():
    # Download and unzip test data into data/meps_example_reduced
    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Unzip(extract_dir=""),
        path="data",
        fname="meps_example_reduced.zip",
    )
    
    return "data/meps_example_reduced"


def test_load_reduced_meps_dataset(ewc_testdata_path):
    datastore = NumpyFilesDatastore(
        root_path=ewc_testdata_path
    )
    datastore.get_xy(category="state", stacked=True)

    datastore.get_dataarray(category="forcing", split="train").unstack("grid_index")
    datastore.get_dataarray(category="state", split="train").unstack("grid_index")

    dataset = WeatherDataset(datastore=datastore)

    var_names = datastore.config.values["dataset"]["var_names"]
    var_units = datastore.config.values["dataset"]["var_units"]
    var_longnames = datastore.config.values["dataset"]["var_longnames"]

    assert len(var_names) == len(var_longnames)
    assert len(var_names) == len(var_units)

    # in future the number of grid static features
    # will be provided by the Dataset class itself
    n_grid_static_features = 4
    # Hardcoded in model
    n_input_steps = 2

    n_forcing_features = datastore.config.values["dataset"]["num_forcing_features"]
    n_state_features = len(var_names)
    n_prediction_timesteps = dataset.ar_steps

    nx, ny = datastore.config.values["grid_shape_state"]
    n_grid = nx * ny

    # check that the dataset is not empty
    assert len(dataset) > 0

    # get the first item
    item = dataset[0]
    init_states = item.init_states
    target_states = item.target_states
    forcing = item.forcing
    
    # check that the shapes of the tensors are correct
    assert init_states.shape == (n_input_steps, n_grid, n_state_features)
    assert target_states.shape == (
        n_prediction_timesteps,
        n_grid,
        n_state_features,
    )
    assert forcing.shape == (
        n_prediction_timesteps,
        n_grid,
        n_forcing_features,
    )
    
    ds_state_norm = datastore.get_normalization_dataarray(category="state")
    
    static_data = {
        "border_mask": datastore.boundary_mask.values,
        "grid_static_features": datastore.get_dataarray(category="static", split="train").values,
        "data_mean": ds_state_norm.state_mean.values,
        "data_std": ds_state_norm.state_std.values,
        "step_diff_mean": ds_state_norm.state_diff_mean.values,
        "step_diff_std": ds_state_norm.state_diff_std.values,
    }

    required_props = {
        "border_mask",
        "grid_static_features",
        "step_diff_mean",
        "step_diff_std",
        "data_mean",
        "data_std",
        "param_weights",
    }

    # check the sizes of the props
    assert static_data["border_mask"].shape == (n_grid, )
    assert static_data["grid_static_features"].shape == (
        n_grid,
        n_grid_static_features,
    )
    assert static_data["step_diff_mean"].shape == (n_state_features,)
    assert static_data["step_diff_std"].shape == (n_state_features,)
    assert static_data["data_mean"].shape == (n_state_features,)
    assert static_data["data_std"].shape == (n_state_features,)
    assert static_data["param_weights"].shape == (n_state_features,)

    assert set(static_data.keys()) == required_props


def test_create_graph_reduced_meps_dataset():
    args = [
        "--graph=hierarchical",
        "--hierarchical=1",
        "--data_config=data/meps_example_reduced/data_config.yaml",
        "--levels=2",
    ]
    create_mesh(args)


def test_train_model_reduced_meps_dataset():
    args = [
        "--model=hi_lam",
        "--data_config=data/meps_example_reduced/data_config.yaml",
        "--n_workers=4",
        "--epochs=1",
        "--graph=hierarchical",
        "--hidden_dim=16",
        "--hidden_layers=1",
        "--processor_layers=1",
        "--ar_steps=1",
        "--eval=val",
        "--n_example_pred=0",
    ]
    train_model(args)
