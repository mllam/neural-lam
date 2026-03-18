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


@dataclasses.dataclass
class DatastoreSelection:
    """
    Configuration for selecting a datastore to use with neural-lam.

    Attributes
    ----------
    kind : str
        The kind of datastore to use, currently `mdp` or `npyfilesmeps` are
        implemented.
    config_path : str
        The path to the configuration file for the selected datastore, this is
        assumed to be relative to the configuration file for neural-lam.
    """

    kind: str

    def __post_init__(self):
        if self.kind not in DATASTORES:
            available = ", ".join(DATASTORES.keys())
            raise ValueError(
                f"Unknown datastore kind '{self.kind}'. "
                f"Available options are: {available}."
            )

    config_path: str


@dataclasses.dataclass
class ManualStateFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    the weights are manually specified.

    Attributes
    ----------
    weights : Dict[str, float]
        Manual weights for the state features.
    """

    weights: Dict[str, float]


@dataclasses.dataclass
class UniformFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    all state features are weighted equally.
    """

    pass


@dataclasses.dataclass
class OutputClamping:
    """
    Configuration for clamping the output of the model.

    Attributes
    ----------
    lower : Dict[str, float]
        The minimum value to clamp each output feature to.
    upper : Dict[str, float]
        The maximum value to clamp each output feature to.
    """

    lower: Dict[str, float] = dataclasses.field(default_factory=dict)
    upper: Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration related to training neural-lam

    Attributes
    ----------
    state_feature_weighting : Union[ManualStateFeatureWeighting,
                                    UniformFeatureWeighting]
        The method to use for weighting the state features in the loss
        function. Defaults to uniform weighting (`UniformFeatureWeighting`, i.e.
        all features are weighted equally).
    """

    state_feature_weighting: Union[
        ManualStateFeatureWeighting, UniformFeatureWeighting
    ] = dataclasses.field(default_factory=UniformFeatureWeighting)

    output_clamping: OutputClamping = dataclasses.field(
        default_factory=OutputClamping
    )


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.JSONWizard, dataclass_wizard.YAMLWizard):
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
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Define the configuration class as a JSON wizard class.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True


class InvalidConfigError(Exception):
    pass


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
    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            "There was an error loading the configuration file at "
            f"{config_path}. "
        ) from ex

    datastore_config_path = (
        Path(config_path).parent / config.datastore.config_path
    )

    datastore = init_datastore(
        datastore_kind=config.datastore.kind,
        config_path=datastore_config_path,
    )

    return config, datastore
