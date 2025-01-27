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
os.environ["WANDB_MODE"] = "disabled"

DATASTORE_EXAMPLES_ROOT_PATH = Path("tests/datastore_examples")

# Initializing variables for the s3 client
S3_BUCKET_NAME = "mllam-testdata"
S3_ENDPOINT_URL = "https://object-store.os-api.cci1.ecmwf.int"
S3_FILE_PATH = "neural-lam/npy/meps_example_reduced.v0.3.0.tar.gz"
S3_FULL_PATH = "/".join([S3_ENDPOINT_URL, S3_BUCKET_NAME, S3_FILE_PATH])
TEST_DATA_KNOWN_HASH = (
    "af16d87099944dda152cc988ce347173da68c56d3739c86ec53c8132981a93e6"
)


def download_meps_example_reduced_dataset():
    # Download and unzip test data into data/meps_example_reduced
    root_path = DATASTORE_EXAMPLES_ROOT_PATH / "npyfilesmeps"
    dataset_path = root_path / "meps_example_reduced"

    pooch.retrieve(
        url=S3_FULL_PATH,
        known_hash=TEST_DATA_KNOWN_HASH,
        processor=pooch.Untar(extract_dir=""),
        path=root_path,
        fname=S3_FILE_PATH.split("/")[-1],
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

DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore


def init_datastore_example(datastore_kind):
    datastore = init_datastore(
        datastore_kind=datastore_kind,
        config_path=DATASTORES_EXAMPLES[datastore_kind],
    )

    return datastore
