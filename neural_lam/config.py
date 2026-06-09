# Standard library
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    Configuration for selecting a datastore and declaring how its variables
    are consumed by the model.

    Attributes
    ----------
    kind : str
        The kind of datastore to use, currently `mdp` or `npyfilesmeps` are
        implemented.
    config_path : str
        The path to the configuration file for the selected datastore, this
        is assumed to be relative to the configuration file for neural-lam.
    inputs : Dict[str, List[str] or None] or None, optional
        Per-category lists of variable names this datastore contributes as
        model inputs. Categories are typically ``state``, ``forcing``,
        ``static``. If the whole field is ``None`` (the default), every
        variable in every category that the datastore exposes is treated
        as an input. If a category key is present with a ``null`` / ``None``
        value, all variables in that category are used. An explicit empty
        list excludes the category.
    outputs : Dict[str, List[str] or None] or None, optional
        Per-category lists of variable names this datastore contributes as
        model outputs (prediction targets). Categories may include ``state``
        for prognostic outputs (those that are fed back as input in the
        next autoregressive step) and ``diagnostic`` for predict-only
        outputs. Prognostic outputs are the intersection of
        ``inputs["state"]`` and ``outputs["state"]``; everything in
        ``outputs`` that is not also in ``inputs["state"]`` is diagnostic.
        If ``None`` (the default), this datastore is treated as input-only
        (no contribution to predictions). ``null`` per-category values
        follow the same "all available" convention as ``inputs``.
    """

    kind: str
    config_path: str
    inputs: Optional[Dict[str, Optional[List[str]]]] = None
    outputs: Optional[Dict[str, Optional[List[str]]]] = None

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
        Mapping from user-chosen datastore name to its selection and role
        declaration. The dict key becomes the canonical source name used
        throughout the pipeline (e.g. as keys in future per-source
        ``ForecastBatch`` field dicts, or to disambiguate weight / clamping
        config when variable names collide between sources).
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


def _validate_output_name_collisions(
    datastores: Dict[str, Union[MDPDatastore, NpyFilesDatastoreMEPS]],
    selections: Dict[str, DatastoreSelection],
) -> None:
    """Raise :class:`InvalidConfigError` if two datastores would contribute
    a variable with the same name to the model's output set, since
    downstream sites (weight dicts, metric keys, saved zarr coords) cannot
    disambiguate.

    Fix is to give the colliding variable a unique name in one of the
    contributing zarrs (mdp's ``dim_mapping.name_format`` for new builds,
    or ``xr.Dataset.assign_coords`` on the small ``{category}_feature``
    coord array of an existing zarr - a milliseconds operation regardless
    of zarr size).
    """
    seen: Dict[str, str] = {}
    for ds_name, sel in selections.items():
        if sel.outputs is None:
            continue
        for category, var_list in sel.outputs.items():
            if var_list is None:
                var_list = datastores[ds_name].get_vars_names(category)
            for var in var_list:
                if var in seen:
                    raise InvalidConfigError(
                        f"Variable '{var}' is declared as an output in "
                        f"both datastores '{seen[var]}' and '{ds_name}'. "
                        "Rename the variable in one of the source zarrs "
                        "(via mdp's `dim_mapping.name_format` for new "
                        "builds, or via `xr.Dataset.assign_coords` on the "
                        "existing zarr's `{category}_feature` coord - a "
                        "milliseconds operation regardless of zarr size). "
                        "See mllam/neural-lam#652."
                    )
                seen[var] = ds_name


def load_config_and_datastore(
    config_path: str,
) -> tuple[
    NeuralLAMConfig,
    Dict[str, Union[MDPDatastore, NpyFilesDatastoreMEPS]],
]:
    """Load the neural-lam configuration and instantiate each datastore.

    The configuration uses the multi-datastore schema introduced for #652:
    a top-level ``datastores`` mapping with one entry per source.

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

    config_dir = Path(config_path).parent
    loaded: Dict[str, Union[MDPDatastore, NpyFilesDatastoreMEPS]] = {}
    for name, selection in config.datastores.items():
        datastore_config_path = config_dir / selection.config_path
        loaded[name] = init_datastore(
            datastore_kind=selection.kind,
            config_path=datastore_config_path,
        )

    _validate_output_name_collisions(loaded, config.datastores)
    return config, loaded
