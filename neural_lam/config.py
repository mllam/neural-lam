"""Configuration dataclasses and helpers for Neural-LAM experiments."""

# Standard library
import dataclasses
from pathlib import Path
from typing import cast

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
    datastore : DatastoreSelection
        Configuration specifying which datastore backend to use and its
        associated settings.
    training : TrainingConfig
        Configuration for training the model, including loss function and
        feature-weighting strategy. Defaults to ``TrainingConfig()``.
    """

    datastore: DatastoreSelection
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
) -> tuple[NeuralLAMConfig, MDPDatastore | NpyFilesDatastoreMEPS]:
    """
    Load the neural-lam configuration and the datastore specified in the
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    tuple[NeuralLAMConfig, MDPDatastore | NpyFilesDatastoreMEPS]
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

    return config, cast(MDPDatastore | NpyFilesDatastoreMEPS, datastore)


@dataclasses.dataclass
class ModelConfig:
    """
    Configuration for neural-lam model architecture.

    Attributes
    ----------
    model : str, default "graph_lam"
        Model architecture name.
    graph : str, default "multiscale"
        Name of the graph to load from the datastore.
    hidden_dim : int, default 64
        Dimensionality of hidden representations.
    hidden_layers : int, default 1
        Number of hidden layers in MLPs.
    processor_layers : int, default 4
        Number of GNN layers in processor GNN.
    mesh_aggr : str, default "sum"
        Aggregation method for m2m GNN layers ("sum" or "mean").
    output_std : bool, default False
        Whether models should output standard deviation per feature.
    g2m_gnn_type : str, default "InteractionNet"
        GNN layer type for grid-to-mesh encoding.
    m2g_gnn_type : str, default "InteractionNet"
        GNN layer type for mesh-to-grid decoding.
    mesh_up_gnn_type : str, default "InteractionNet"
        GNN layer type for upward mesh message passing.
    mesh_down_gnn_type : str, default "InteractionNet"
        GNN layer type for downward mesh message passing.
    """

    model: str = "graph_lam"
    graph: str = "multiscale"
    hidden_dim: int = 64
    hidden_layers: int = 1
    processor_layers: int = 4
    mesh_aggr: str = "sum"
    output_std: bool = False
    g2m_gnn_type: str = "InteractionNet"
    m2g_gnn_type: str = "InteractionNet"
    mesh_up_gnn_type: str = "InteractionNet"
    mesh_down_gnn_type: str = "InteractionNet"


@dataclasses.dataclass
class TrainRunConfig:
    """
    Hyperparameters and settings for a training or evaluation run.

    Attributes
    ----------
    epochs : int, default 200
        Number of training epochs.
    batch_size : int, default 4
        Batch size.
    ar_steps_train : int, default 1
        Autoregressive rollout steps during training.
    ar_steps_eval : int, default 10
        Autoregressive rollout steps during evaluation.
    loss : str, default "wmse"
        Loss function to use ("mse" or "wmse").
    lr : float, default 1e-3
        Learning rate.
    val_interval : int, default 1
        Validation epoch frequency.
    num_sanity_val_steps : int, default 2
        Number of validation batches to run prior to training.
    val_steps_to_log : list of int, default [1]
        Rollout steps to log during validation.
    train_steps_to_log : list of int, default []
        Rollout steps to log during training.
    metrics_watch : list of str, default []
        List of watched metrics.
    var_leads_metrics_watch : dict of {int: list of int}, default {}
        Mapping from variable index to list of watched rollout steps.
    load : str or None, default None
        Checkpoint path to load.
    restore_opt : bool, default False
        Whether to restore optimizer state from checkpoint.
    eval : str or None, default None
        Evaluation split name ("val", "test") or None if training.
    n_example_pred : int, default 1
        Number of example predictions to plot during testing.
    create_gif : bool, default False
        Whether to create animation GIFs during evaluation.
    """

    epochs: int = 200
    batch_size: int = 4
    ar_steps_train: int = 1
    ar_steps_eval: int = 10
    loss: str = "wmse"
    lr: float = 1e-3
    val_interval: int = 1
    num_sanity_val_steps: int = 2
    val_steps_to_log: list[int] = dataclasses.field(default_factory=lambda: [1])
    train_steps_to_log: list[int] = dataclasses.field(default_factory=list)
    metrics_watch: list[str] = dataclasses.field(default_factory=list)
    var_leads_metrics_watch: dict[int, list[int]] = dataclasses.field(
        default_factory=dict
    )
    load: str | None = None
    restore_opt: bool = False
    eval: str | None = None
    n_example_pred: int = 1
    create_gif: bool = False


@dataclasses.dataclass
class DataConfig:
    """
    Data loading and forcing parameters.

    Attributes
    ----------
    num_past_forcing_steps : int, default 1
        Number of past forcing time steps as input.
    num_future_forcing_steps : int, default 1
        Number of future forcing time steps as input.
    num_workers : int, default 4
        Number of data loader workers.
    load_single_member : bool, default False
        Whether to load only ensemble member 0.
    """

    num_past_forcing_steps: int = 1
    num_future_forcing_steps: int = 1
    num_workers: int = 4
    load_single_member: bool = False


@dataclasses.dataclass
class ComputeConfig:
    """
    Compute and accelerator configuration.

    Attributes
    ----------
    seed : int, default 42
        Random seed.
    num_nodes : int, default 1
        Number of cluster nodes.
    devices : str or list of int, default "auto"
        Hardware devices to use.
    precision : str or int, default 32
        Floating point precision.
    """

    seed: int = 42
    num_nodes: int = 1
    devices: str | list[int] = "auto"
    precision: str | int = 32


@dataclasses.dataclass
class LoggingConfig:
    """
    Experiment tracking and artifact logging configuration.

    Attributes
    ----------
    logger : str, default "wandb"
        Logger backend name ("wandb" or "mlflow").
    logger_project : str, default "neural_lam"
        Project name for the logger.
    logger_run_name : str or None, default None
        Custom run name for the logger.
    runs_root : str, default "runs"
        Root directory for run artifacts and checkpoints.
    wandb_id : str or None, default None
        W&B run ID for resuming experiments.
    """

    logger: str = "wandb"
    logger_project: str = "neural_lam"
    logger_run_name: str | None = None
    runs_root: str = "runs"
    wandb_id: str | None = None


try:
    # Standard library
    import argparse

    # Third-party
    import torch

    torch.serialization.add_safe_globals(
        [
            NeuralLAMConfig,
            DatastoreSelection,
            TrainingConfig,
            OutputClamping,
            UniformFeatureWeighting,
            ModelConfig,
            TrainRunConfig,
            DataConfig,
            ComputeConfig,
            LoggingConfig,
            argparse.Namespace,
        ]
    )
except (ImportError, AttributeError):
    pass
