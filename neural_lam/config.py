"""Configuration dataclasses and helpers for Neural-LAM experiments."""

# Standard library
import dataclasses
from pathlib import Path
from typing import cast

# Third-party
import dataclass_wizard
import yaml

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
    config_path: str

    def __post_init__(self) -> None:
        """
        Validate that the selected datastore kind is implemented.

        Raises
        ------
        ValueError
            If the provided ``kind`` is not part of :data:`DATASTORES`.
        """
        if self.kind not in DATASTORES:
            raise ValueError(f"Datastore kind {self.kind} is not implemented")


@dataclasses.dataclass
class ManualStateFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    the weights are manually specified.

    Attributes
    ----------
    weights : dict[str, float]
        Manual weights for the state features.
    """

    weights: dict[str, float]


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
    lower : dict[str, float]
        The minimum value to clamp each output feature to.
    upper : dict[str, float]
        The maximum value to clamp each output feature to.
    """

    lower: dict[str, float] = dataclasses.field(default_factory=dict)
    upper: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration related to training neural-lam

    Attributes
    ----------
    state_feature_weighting :
        ManualStateFeatureWeighting | UniformFeatureWeighting
        The method to use for weighting the state features in the loss
        function. Defaults to uniform weighting (`UniformFeatureWeighting`, i.e.
        all features are weighted equally).
    output_clamping : OutputClamping
        Per-feature lower / upper clamping bounds applied to the model output.
        Defaults to an empty ``OutputClamping`` (no clamping).
    """

    state_feature_weighting: (
        ManualStateFeatureWeighting | UniformFeatureWeighting
    ) = dataclasses.field(default_factory=UniformFeatureWeighting)

    output_clamping: OutputClamping = dataclasses.field(
        default_factory=OutputClamping
    )


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.JSONWizard, dataclass_wizard.YAMLWizard):
    """
    Configuration for the Neural-LAM model and training pipeline.

    Loads and stores all settings needed to run Neural-LAM, including
    datastore selection and training hyperparameters. Serialisation and
    deserialisation from YAML/JSON is handled via ``dataclass_wizard``.

    Attributes
    ----------
    datastores : dict[str, DatastoreSelection]
        Mapping from a user-chosen datastore name to its selection config.
        A datastore's role follows from the categories it provides: one with
        `state` data is the interior (input and output), one without is
        input-only (boundary forcing). :func:`load_config_and_datastore`
        requires exactly one interior and at most one boundary.
    training : TrainingConfig
        Configuration for training the model, including loss function and
        feature-weighting strategy. Defaults to ``TrainingConfig()``.
    """

    datastores: dict[str, DatastoreSelection]
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Define the configuration class as a JSON wizard class.

        Together `tag_key` and `auto_assign_tags` enable that when a `Union` of
        types are used for an attribute, the specific type to deserialize to
        can be specified in the serialised data using the `tag_key` value. In
        our case we call the tag key `__config_class__` to indicate to the
        user that they should pick a dataclass describing configuration in
        neural-lam. This Union-based selection allows us to support different
        configuration attributes for different choices of methods for example
        and is used when picking between different feature weighting methods in
        the `TrainingConfig` class. `auto_assign_tags` is set to True to
        automatically set that tag key (i.e. `__config_class__` in the config
        file) should just be the class name of the dataclass to deserialize to.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True
        # ensure that all parts of the loaded configuration match the
        # dataclasses used
        # TODO: this should be enabled once
        # https://github.com/rnag/dataclass-wizard/issues/137 is fixed, but
        # currently cannot be used together with `auto_assign_tags` due to a
        # bug it seems
        # raise_on_unknown_json_key = True


class InvalidConfigError(Exception):
    """Raised when the Neural-LAM configuration file is invalid or malformed."""

    pass


def load_config_and_datastore(
    config_path: str,
) -> tuple[
    NeuralLAMConfig,
    MDPDatastore | NpyFilesDatastoreMEPS,
    MDPDatastore | NpyFilesDatastoreMEPS | None,
]:
    """
    Load the neural-lam configuration and the datastores specified in the
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    tuple[NeuralLAMConfig, datastore, datastore_boundary]
        The configuration, the interior datastore (the one with `state`
        data), and the boundary datastore (the one without, or None).

    Raises
    ------
    InvalidConfigError
        If not exactly one datastore provides `state` data, or if more than
        one omits it.
    """
    with open(config_path, encoding="utf-8") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as ex:
            raise InvalidConfigError(
                f"Could not parse the configuration file at {config_path}: "
                f"{ex}"
            ) from ex
    if isinstance(raw_config, dict) and (
        "datastore" in raw_config and "datastores" not in raw_config
    ):
        raise InvalidConfigError(
            "The `datastore:` config key has been replaced by a named "
            "`datastores:` mapping (the role of each datastore is now implied "
            "by the categories it provides). Move your datastore under a "
            "named entry, e.g.:\n"
            "datastores:\n"
            "  <name>:\n"
            "    kind: ...\n"
            "    config_path: ...\n"
            "See the README and CHANGELOG for details."
        )

    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            "There was an error loading the configuration file at "
            f"{config_path}. "
        ) from ex

    # Datastore config paths are relative to the neural-lam config file.
    config_dir = Path(config_path).parent
    interior_datastores = {}
    boundary_datastores = {}
    for name, selection in config.datastores.items():
        datastore = init_datastore(
            datastore_kind=selection.kind,
            config_path=config_dir / selection.config_path,
        )
        if datastore.get_num_data_vars(category="state") > 0:
            interior_datastores[name] = datastore
        else:
            boundary_datastores[name] = datastore

    if len(interior_datastores) != 1:
        raise InvalidConfigError(
            "Exactly one datastore must provide `state` data (the interior "
            f"domain), but {len(interior_datastores)} were found in "
            f"{config_path}: {sorted(interior_datastores)}."
        )
    if len(boundary_datastores) > 1:
        raise InvalidConfigError(
            "At most one boundary datastore (a datastore without `state` "
            f"data) is currently supported, but {len(boundary_datastores)} "
            f"were found in {config_path}: {sorted(boundary_datastores)}."
        )

    (datastore,) = interior_datastores.values()
    datastore_boundary = next(iter(boundary_datastores.values()), None)

    return (
        config,
        cast(MDPDatastore | NpyFilesDatastoreMEPS, datastore),
        cast(MDPDatastore | NpyFilesDatastoreMEPS | None, datastore_boundary),
    )
