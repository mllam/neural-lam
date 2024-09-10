# Standard library
import os
from pathlib import Path

# Third-party
import pooch
import yaml

# First-party
from neural_lam.datastore import mllam, npyfiles

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_DISABLED"] = "true"

DATASTORES = dict(
    mllam=mllam.MLLAMDatastore,
    npyfiles=npyfiles.NpyFilesDatastore,
)

DATASTORE_EXAMPLES_ROOT_PATH = Path("tests/datastore_examples")

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
    root_path = DATASTORE_EXAMPLES_ROOT_PATH / "npy"
    dataset_path = root_path / "meps_example_reduced"

    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Unzip(extract_dir=""),
        path=root_path,
        fname="meps_example_reduced.zip",
    )

    config_path = dataset_path / "data_config.yaml"

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

    return config_path


DATASTORES_EXAMPLES = dict(
    mllam=dict(
        config_path=DATASTORE_EXAMPLES_ROOT_PATH
        / "mllam"
        / "danra.example.yaml"
    ),
    npyfiles=dict(config_path=download_meps_example_reduced_dataset()),
)


def init_datastore(datastore_name):
    DatastoreClass = DATASTORES[datastore_name]
    return DatastoreClass(**DATASTORES_EXAMPLES[datastore_name])
