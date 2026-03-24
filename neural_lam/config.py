# Standard library
import dataclasses
import argparse
from pathlib import Path
from typing import Dict, Union, Tuple

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

    Args:
        kind (str): The kind of datastore to use. Currently 'mdp' or 
            'npyfilesmeps' are implemented.
        config_path (str): The path to the configuration file for the selected 
            datastore, assumed to be relative to the neural-lam config file.
    """

    kind: str

    def __post_init__(self):
        """
        Validates the datastore kind against registered DATASTORES.

        Raises:
            ValueError: If the provided kind is not found in the DATASTORES registry.
        """
        if self.kind not in DATASTORES:
            available = ", ".join(DATASTORES.keys())
            raise ValueError(
                f"Unknown datastore kind '{self.kind}'. "
                f"Supported options are: {available}. "
                "Please verify your configuration file."
            )

    config_path: str


@dataclasses.dataclass
class ManualStateFeatureWeighting:
    """
    Configuration for manual weighting of state features in the loss function.

    Args:
        weights (Dict[str, float]): Dictionary mapping feature names to 
            their respective manual weights.
    """

    weights: Dict[str, float]


@dataclasses.dataclass
class UniformFeatureWeighting:
    """
    Configuration for equal weighting of all state features in the loss function.
    """

    pass


@dataclasses.dataclass
class OutputClamping:
    """
    Configuration for clamping the model's output values.

    Args:
        lower (Dict[str, float]): Minimum values for each output feature.
        upper (Dict[str, float]): Maximum values for each output feature.
    """

    lower: Dict[str, float] = dataclasses.field(default_factory=dict)
    upper: Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration parameters related to the training process of neural-lam.

    Args:
        state_feature_weighting (Union[ManualStateFeatureWeighting, UniformFeatureWeighting]): 
            The method used for weighting state features. Defaults to 
            UniformFeatureWeighting.
        output_clamping (OutputClamping): Clamping configuration for model 
            predictions.
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
    Primary configuration class for Neural-LAM. Handles loading and 
    storing all parameters required for model execution and training.

    Args:
        datastore (DatastoreSelection): Selection and config path for the data source.
        training (TrainingConfig): Training-specific parameters and loss weighting.
    """

    datastore: DatastoreSelection
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Metadata for the JSON/YAML Wizard to handle configuration tagging.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True


class InvalidConfigError(Exception):
    """Raised when the configuration file contains invalid keys or structure."""
    pass


def load_config_and_datastore(
    config_path: str,
) -> Tuple[NeuralLAMConfig, Union[MDPDatastore, NpyFilesDatastoreMEPS]]:
    """
    Loads the Neural-LAM configuration and initializes the specified datastore.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Tuple[NeuralLAMConfig, Union[MDPDatastore, NpyFilesDatastoreMEPS]]: 
            A tuple containing the validated configuration object and the 
            initialized datastore instance.

    Raises:
        InvalidConfigError: If the configuration file is missing required keys 
            or has an invalid structure.
        FileNotFoundError: If the config_path does not exist.
    """
    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            f"Failed to load configuration at '{config_path}'. "
            "Ensure all keys match the NeuralLAMConfig schema."
        ) from ex

    # Resolve datastore path relative to the main config file
    datastore_config_path = (
        Path(config_path).parent / config.datastore.config_path
    )

    datastore = init_datastore(
        datastore_kind=config.datastore.kind,
        config_path=datastore_config_path,
    )

    return config, datastore