"""Configuration dataclasses and helpers for Neural-LAM experiments."""

# Standard library
import dataclasses
from pathlib import Path
from typing import Dict, Optional, Union

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
    config_path: str

    def __post_init__(self):
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
    output_clamping : OutputClamping
        Per-feature lower / upper clamping bounds applied to the model output.
        Defaults to an empty ``OutputClamping`` (no clamping).
    """

    state_feature_weighting: Union[
        ManualStateFeatureWeighting, UniformFeatureWeighting
    ] = dataclasses.field(default_factory=UniformFeatureWeighting)

    output_clamping: OutputClamping = dataclasses.field(
        default_factory=OutputClamping
    )


@dataclasses.dataclass
class ProbabilisticConfig:
    """
    Configuration for the probabilistic (Graph-EFM) models.

    Only used when training or evaluating a probabilistic model
    (``--model graph_efm``/``graph_efm_ms``); ignored otherwise. Every field
    has a default, so the ``probabilistic`` config section may be omitted
    entirely for deterministic models.

    Attributes
    ----------
    latent_dim : int, optional
        Dimensionality of the latent variable at each latent-carrying mesh
        node. Defaults to the model's ``hidden_dim`` when None.
    prior_layers : int
        Number of on-mesh GNN layers in the prior.
    encoder_layers : int
        Number of on-mesh GNN layers in the variational encoder.
    decoder_layers : int
        Number of on-mesh GNN layers in the latent decoder.
    learn_prior : bool
        If True, the prior is a learned encoder conditioned on the previous
        state; if False, a constant ``Normal(0, 1)`` prior is used.
    prior_dist : str
        Output distribution of the prior: ``"isotropic"`` or ``"diagonal"``.
    kl_beta : float
        Weight of the KL term in the ELBO. When 0, the prior and KL are not
        computed (pure auto-encoder training).
    eval_ensemble_size : int
        Number of ensemble members sampled during validation and testing.
    """

    latent_dim: Optional[int] = None
    prior_layers: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 4
    learn_prior: bool = True
    prior_dist: str = "isotropic"
    kl_beta: float = 1.0
    eval_ensemble_size: int = 5


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.JSONWizard, dataclass_wizard.YAMLWizard):
    """
    Configuration for the Neural-LAM model and training pipeline.

    Loads and stores all settings needed to run Neural-LAM, including
    datastore selection and training hyperparameters. Serialisation and
    deserialisation from YAML/JSON is handled via ``dataclass_wizard``.

    Attributes
    ----------
    datastore : DatastoreSelection
        Configuration specifying which datastore backend to use and its
        associated settings.
    training : TrainingConfig
        Configuration for training the model, including loss function and
        feature-weighting strategy. Defaults to ``TrainingConfig()``.
    probabilistic : ProbabilisticConfig
        Configuration for the probabilistic (Graph-EFM) models. Defaults to
        ``ProbabilisticConfig()`` and is ignored by deterministic models.
    """

    datastore: DatastoreSelection
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    probabilistic: ProbabilisticConfig = dataclasses.field(
        default_factory=ProbabilisticConfig
    )

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
    # datastore config is assumed to be relative to the config file
    datastore_config_path = (
        Path(config_path).parent / config.datastore.config_path
    )
    datastore = init_datastore(
        datastore_kind=config.datastore.kind, config_path=datastore_config_path
    )

    return config, datastore
