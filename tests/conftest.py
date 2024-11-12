# Standard library
import os
from pathlib import Path

# Third-party
import pooch
import yaml

# First-party
from neural_lam.datastore import DATASTORES, init_datastore
from neural_lam.datastore.npyfilesmeps import (
    compute_standardization_stats as compute_standardization_stats_meps,
)

# Local
from .dummy_datastore import DummyDatastore

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_DISABLED"] = "true"

DATASTORE_EXAMPLES_ROOT_PATH = Path("tests/datastore_examples")

# Initializing variables for the s3 client
S3_BUCKET_NAME = "mllam-testdata"
# S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
S3_ENDPOINT_URL = "http://localhost:8000"
# S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.1.0.zip"
# TODO: I will upload this to AWS S3 once I have located the credentials...
S3_FILE_PATH = "meps_example_reduced.v0.2.0.zip"
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

    # create parameters
    compute_standardization_stats_meps.main(
        datastore_config_path=config_path,
        batch_size=8,
        step_length=3,
        n_workers=1,
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

DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore


def init_datastore_example(datastore_kind):
    datastore = init_datastore(
        datastore_kind=datastore_kind,
        config_path=DATASTORES_EXAMPLES[datastore_kind],
    )

    return datastore
