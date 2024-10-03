# Standard library
import dataclasses
from pathlib import Path
from typing import Dict, Union

# Third-party
import dataclass_wizard

# Local
from .datastore import (
    DATASTORES,
    MDPDatastore,
    NpyFilesDatastoreMEPS,
    init_datastore,
)


class DatastoreKindStr(str):
    VALID_KINDS = DATASTORES.keys()

    def __new__(cls, value):
        if value not in cls.VALID_KINDS:
            raise ValueError(f"Invalid datastore kind: {value}")
        return super().__new__(cls, value)


@dataclasses.dataclass
class DatastoreSelection:
    """
    Configuration for selecting a datastore to use with neural-lam.

    Attributes
    ----------
    kind : DatastoreKindStr
        The kind of datastore to use, currently `mdp` or `npyfilesmeps` are
        implemented.
    config_path : str
        The path to the configuration file for the selected datastore, this is
        assumed to be relative to the configuration file for neural-lam.
    """

    kind: DatastoreKindStr
    config_path: str


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration related to training neural-lam

    Attributes
    ----------
    state_feature_weights : Dict[str, float]
        The weights for each state feature in the datastore to use in the loss
        function during training.
    """

    state_feature_weights: Dict[str, float]


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.YAMLWizard):
    """
    Dataclass for Neural-LAM configuration. This class is used to load and
    store the configuration for using Neural-LAM.

    Attributes
    ----------
    datastore : DatastoreSelection
        The configuration for the datastore to use.
    training : TrainingConfig
        The configuration for training the model.
    """

    datastore: DatastoreSelection
    training: TrainingConfig


def load_config_and_datastore(
    config_path: str,
) -> tuple[NeuralLAMConfig, Union[MDPDatastore, NpyFilesDatastoreMEPS]]:
    """
    Load the neural-lam configuration and the datastore specified in the
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    tuple[NeuralLAMConfig, Union[MDPDatastore, NpyFilesDatastoreMEPS]]
        The Neural-LAM configuration and the loaded datastore.
    """
    config = NeuralLAMConfig.from_yaml_file(config_path)
    # datastore config is assumed to be relative to the config file
    datastore_config_path = (
        Path(config_path).parent / config.datastore.config_path
    )
    datastore = init_datastore(
        datastore_kind=config.datastore.kind, config_path=datastore_config_path
    )

    # TODO: This check should maybe be moved somewhere else, but I'm not sure
    # where right now... check that the config state feature weights include a
    # weight for each state feature
    state_feature_names = datastore.get_vars_names(category="state")
    named_feature_weights = config.training.state_feature_weights.keys()

    if set(named_feature_weights) != set(state_feature_names):
        additional_features = set(named_feature_weights) - set(
            state_feature_names
        )
        missing_features = set(state_feature_names) - set(named_feature_weights)
        raise ValueError(
            f"State feature weights must be provided for each state feature in "
            f"the datastore ({state_feature_names}). {missing_features} are "
            "missing and weights are defined for the features "
            f"{additional_features} which are not in the datastore."
        )

    return config, datastore
