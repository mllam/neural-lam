# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pooch
import torch
import yaml

# First-party
from neural_lam.datastore import DATASTORES, init_datastore
from neural_lam.datastore.npyfilesmeps import (
    compute_standardization_stats as compute_standardization_stats_meps,
)
from neural_lam.utils import get_stacked_xy

# Local
from .dummy_datastore import DummyDatastore

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_MODE"] = "disabled"

DATASTORE_EXAMPLES_ROOT_PATH = Path("tests/datastore_examples")

# Initializing variables for the s3 client
S3_BUCKET_NAME = "mllam-testdata"
S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.2.0.zip"
S3_FULL_PATH = "/".join([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_FILE_PATH])
TEST_DATA_KNOWN_HASH = (
    "7ff2e07e04cfcd77631115f800c9d49188bb2a7c2a2777da3cea219f926d0c86"
)


def download_meps_example_reduced_dataset():
    # Download and unzip test data into data/meps_example_reduced
    root_path = DATASTORE_EXAMPLES_ROOT_PATH / "npyfilesmeps"
    dataset_path = root_path / "meps_example_reduced"

    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Unzip(extract_dir=""),
        path=root_path,
        fname="meps_example_reduced.zip",
    )

    config_path = dataset_path / "meps_example_reduced.datastore.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "class" in config["projection"]:
        # XXX: should update the dataset stored on S3 with the change below
        #
        # rename the `projection.class` key to `projection.class_name` in the
        # config this is because the `class` key is reserved for the class
        # attribute of the object and so we can't use it to define a python
        # dataclass
        config["projection"]["class_name"] = config["projection"].pop("class")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    # create parameters, only run if the files we expect are not present
    expected_parameter_files = [
        "parameter_mean.pt",
        "parameter_std.pt",
        "diff_mean.pt",
        "diff_std.pt",
    ]
    expected_parameter_filepaths = [
        dataset_path / "static" / fn for fn in expected_parameter_files
    ]
    if any(not p.exists() for p in expected_parameter_filepaths):
        compute_standardization_stats_meps.main(
            datastore_config_path=config_path,
            batch_size=8,
            step_length=3,
            n_workers=0,
            distributed=False,
        )

    return config_path


DATASTORES_EXAMPLES = dict(
    mdp=(
        DATASTORE_EXAMPLES_ROOT_PATH
        / "mdp"
        / "danra_100m_winds"
        / "danra.datastore.yaml"
    ),
    npyfilesmeps=download_meps_example_reduced_dataset(),
    dummydata=None,
)

DATASTORES_BOUNDARY_EXAMPLES = {
    "mdp": (
        DATASTORE_EXAMPLES_ROOT_PATH
        / "mdp"
        / "era5_1000hPa_danra_100m_winds"
        / "era5.datastore.yaml"
    ),
}

DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore


def init_datastore_example(datastore_kind):
    datastore = init_datastore(
        datastore_kind=datastore_kind,
        config_path=DATASTORES_EXAMPLES[datastore_kind],
    )
    return datastore


def init_datastore_boundary_example(datastore_kind):
    datastore_boundary = init_datastore(
        datastore_kind=datastore_kind,
        config_path=DATASTORES_BOUNDARY_EXAMPLES[datastore_kind],
    )

    return datastore_boundary


def get_test_mesh_dist(datastore, datastore_boundary):
    """Compute a good mesh_node_distance for testing graph creation with
    given datastores
    """
    xy = get_stacked_xy(datastore, datastore_boundary)  # (num_grid, 2)
    # Compute minimum coordinate extent
    min_extent = min(np.ptp(xy, axis=0))

    # Want at least 10 mesh nodes in each direction
    return min_extent / 10.0


def check_saved_graph(graph_dir_path, hierarchical, num_levels=1):
    """Perform all checking for a saved graph"""
    required_graph_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "m2m_node_features.pt",
    ]

    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )

    # TODO: check that the number of edges is consistent over the files, for
    # now we just check the number of features
    d_features = 3
    d_mesh_static = 2

    assert graph_dir_path.exists()

    # check that all the required files are present
    for file_name in required_graph_files:
        assert (graph_dir_path / file_name).exists()

    # try to load each and ensure they have the right shape
    for file_name in required_graph_files:
        file_id = Path(file_name).stem  # remove the extension
        result = torch.load(graph_dir_path / file_name, weights_only=True)

        if file_id.startswith("g2m") or file_id.startswith("m2g"):
            assert isinstance(result, torch.Tensor)

            if file_id.endswith("_index"):
                assert result.shape[0] == 2  # adjacency matrix uses two rows
            elif file_id.endswith("_features"):
                assert result.shape[1] == d_features

        elif file_id.startswith("m2m") or file_id.startswith("mesh"):
            assert isinstance(result, list)
            if not hierarchical:
                assert len(result) == 1
            else:
                if file_id.startswith("mesh_up") or file_id.startswith(
                    "mesh_down"
                ):
                    assert len(result) == num_levels - 1
                else:
                    assert len(result) == num_levels

            for r in result:
                assert isinstance(r, torch.Tensor)

                if file_id == "m2m_node_features":
                    assert r.shape[1] == d_mesh_static
                elif file_id.endswith("_index"):
                    assert r.shape[0] == 2  # adjacency matrix uses two rows
                elif file_id.endswith("_features"):
                    assert r.shape[1] == d_features
