# Standard library
import os

# Third-party
import pooch

# First-party
from create_mesh import main as create_mesh
from neural_lam.config import Config
from neural_lam.utils import load_static_data
from neural_lam.weather_dataset import WeatherDataset
from train_model import main as train_model

os.environ["WANDB_DISABLED"] = "true"


def test_retrieve_data_ewc():
    # Initializing variables for the client
    S3_BUCKET_NAME = "mllam-testdata"
    S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
    S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.1.0.zip"
    S3_FULL_PATH = "/".join([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_FILE_PATH])
    known_hash = (
        "7d80f0d8c3022aa8c0331f26a17566b44b4b33a5d9a60f6d2e60bf65ed857d86"
    )

    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=known_hash,
        processor=pooch.Unzip(extract_dir=""),
        path="data",
        fname="meps_example_reduced.zip",
    )


def test_load_reduced_meps_dataset():
    data_config_file = "data/meps_example_reduced/data_config.yaml"
    dataset_name = "meps_example_reduced"

    dataset = WeatherDataset(dataset_name="meps_example_reduced")
    config = Config.from_file(data_config_file)

    var_names = config.values["dataset"]["var_names"]
    var_units = config.values["dataset"]["var_units"]
    var_longnames = config.values["dataset"]["var_longnames"]

    assert len(var_names) == len(var_longnames)
    assert len(var_names) == len(var_units)

    # TODO: can these two variables be loaded from elsewhere?
    n_grid_static_features = 4
    n_input_steps = 2

    n_forcing_features = config.values["dataset"]["num_forcing_features"]
    n_state_features = len(var_names)
    n_prediction_timesteps = dataset.sample_length - n_input_steps

    nx, ny = config.values["grid_shape_state"]
    n_grid = nx * ny

    # check that the dataset is not empty
    assert len(dataset) > 0

    # get the first item
    init_states, target_states, forcing = dataset[0]

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

    static_data = load_static_data(dataset_name=dataset_name)

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
    assert static_data["border_mask"].shape == (n_grid, 1)
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
