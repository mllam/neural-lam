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
        The path to the configuration file for the selected datastore, this
        is assumed to be relative to the configuration file for neural-lam.
    """

    kind: str
    config_path: str

    def __post_init__(self):
        if self.kind not in DATASTORES:
            raise ValueError(f"Datastore kind {self.kind} is not implemented")


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
        function. Defaults to uniform weighting (`UniformFeatureWeighting`,
        i.e. all features are weighted equally).
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
    datastores : Dict[str, DatastoreSelection]
        Mapping from user-chosen datastore name to its selection. The dict
        key becomes the canonical source name used throughout the pipeline.
        This PR ships only the dict shape; multi-source support (more than
        one entry) lands together with the per-category `inputs`/`outputs`
        filtering follow-up - see mllam/neural-lam#652.
    training : TrainingConfig
        The configuration for training the model.
    """

    datastores: Dict[str, DatastoreSelection]
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Define the configuration class as a JSON wizard class.

        Together `tag_key` and `auto_assign_tags` enable that when a `Union`
        of types are used for an attribute, the specific type to deserialize
        to can be specified in the serialised data using the `tag_key`
        value. In our case we call the tag key `__config_class__` to
        indicate to the user that they should pick a dataclass describing
        configuration in neural-lam. This Union-based selection allows us
        to support different configuration attributes for different choices
        of methods for example and is used when picking between different
        feature weighting methods in the `TrainingConfig` class.
        `auto_assign_tags` is set to True to automatically set that tag key
        (i.e. `__config_class__` in the config file) should just be the
        class name of the dataclass to deserialize to.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True
        # ensure that all parts of the loaded configuration match the
        # dataclasses used
        # TODO: this should be enabled once
        # https://github.com/rnag/dataclass-wizard/issues/137 is fixed, but
        # currently cannot be used together with `auto_assign_tags` due to
        # a bug it seems
        # raise_on_unknown_json_key = True


class InvalidConfigError(Exception):
    pass


def load_config_and_datastore(
    config_path: str,
) -> tuple[
    NeuralLAMConfig,
    Dict[str, Union[MDPDatastore, NpyFilesDatastoreMEPS]],
]:
    """Load the neural-lam configuration and instantiate each datastore.

    The configuration uses the multi-datastore schema introduced for #652:
    a top-level ``datastores`` mapping with one entry per source. This PR
    accepts the dict shape but enforces exactly one entry; multi-source
    support lands together with per-category variable filtering in a
    follow-up.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    config : NeuralLAMConfig
        The parsed configuration.
    datastores : Dict[str, BaseDatastore]
        Mapping from each user-chosen datastore name to the loaded
        datastore object, in the same order as declared in the config.
    """
    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            "There was an error loading the configuration file at "
            f"{config_path}. "
        ) from ex

    if not config.datastores:
        raise InvalidConfigError(
            f"Configuration at {config_path} declares no datastores. "
            "Add at least one entry under the top-level `datastores:` key."
        )
    if len(config.datastores) != 1:
        raise InvalidConfigError(
            "This release accepts exactly one datastore under "
            "`datastores:`. Multi-source support lands together with the "
            "per-category `inputs`/`outputs` filtering follow-up "
            "(see mllam/neural-lam#652)."
        )

    config_dir = Path(config_path).parent
    loaded: Dict[str, Union[MDPDatastore, NpyFilesDatastoreMEPS]] = {}
    for name, selection in config.datastores.items():
        datastore_config_path = config_dir / selection.config_path
        loaded[name] = init_datastore(
            datastore_kind=selection.kind,
            config_path=datastore_config_path,
        )

    return config, loaded
