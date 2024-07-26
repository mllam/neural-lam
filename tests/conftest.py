# Standard library
import os
from pathlib import Path

# Third-party
import pooch
import yaml

# First-party
from neural_lam.datastore.mllam import MLLAMDatastore
from neural_lam.datastore.multizarr import MultiZarrDatastore
from neural_lam.datastore.npyfiles import NpyFilesDatastore

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_DISABLED"] = "true"

DATASTORES = dict(
    multizarr=MultiZarrDatastore,
    mllam=MLLAMDatastore,
    npyfiles=NpyFilesDatastore,
)

# Initializing variables for the s3 client
S3_BUCKET_NAME = "mllam-testdata"
S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.1.0.zip"
S3_FULL_PATH = "/".join([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_FILE_PATH])
TEST_DATA_KNOWN_HASH = (
    "98c7a2f442922de40c6891fe3e5d190346889d6e0e97550170a82a7ce58a72b7"
)


def download_meps_example_reduced_dataset():
    # Download and unzip test data into data/meps_example_reduced
    root_path = Path("tests/datastore_configs/npy")
    dataset_path = root_path / "meps_example_reduced"
    will_download = not dataset_path.exists()

    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Unzip(extract_dir=""),
        path=root_path,
        fname="meps_example_reduced.zip",
    )

    if will_download:
        # XXX: should update the dataset stored on S3 the change below
        config_path = dataset_path / "data_config.yaml"
        # rename the `projection.class` key to `projection.class_name` in the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config["projection.class_name"] = config.pop("projection.class")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    return dataset_path


DATASTORES_EXAMPLES = dict(
    multizarr=dict(root_path="tests/datastore_configs/multizarr"),
    mllam=dict(root_path="tests/datastore_configs/mllam"),
    npyfiles=dict(root_path=download_meps_example_reduced_dataset()),
)


def init_datastore(datastore_name):
    DatastoreClass = DATASTORES[datastore_name]
    return DatastoreClass(**DATASTORES_EXAMPLES[datastore_name])
