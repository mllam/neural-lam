# Standard library
import os
from pathlib import Path

# Third-party
import pooch

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
    root_path = Path("tests/datastores_examples/npy")
    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Unzip(extract_dir=""),
        path=root_path,
        fname="meps_example_reduced.zip",
    )
    return root_path / "meps_example_reduced"


DATASTORES_EXAMPLES = dict(
    multizarr=dict(
        config_path="tests/datastore_configs/multizarr/data_config.yaml"
    ),
    mllam=dict(config_path="tests/datastore_configs/mllam/example.danra.yaml"),
    npyfiles=dict(root_path=download_meps_example_reduced_dataset()),
)


def init_datastore(datastore_name):
    DatastoreClass = DATASTORES[datastore_name]
    return DatastoreClass(**DATASTORES_EXAMPLES[datastore_name])
