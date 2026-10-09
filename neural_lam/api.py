"""High-level Python API for Neural-LAM."""

# Standard library
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Local
from . import create_graph as create_graph_script
from . import train_model as train_model_script
from .config import (
    ComputeConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    NeuralLAMConfig,
    TrainRunConfig,
    load_config_and_datastore,
)
from .datastore.base import BaseDatastore


@dataclass
class Run:
    """Handle containing output paths for a Neural-LAM run."""

    run_dir: Path
    checkpoint_path: Path | None = None

    @property
    def plot_dir(self) -> Path:
        """Directory containing generated plots."""
        return self.run_dir / "plots"

    @property
    def example_plots(self) -> Path:
        """Directory containing example evaluation plots."""
        return self.plot_dir / "example_plots"

    @property
    def rmse_plot(self) -> Path:
        """Path to the generated RMSE map plot."""
        return self.plot_dir / "rmse.png"


def train(
    *,
    config_path: str | None = None,
    model: str = "graph_lam",
    seed: int = 42,
    num_workers: int = 4,
    num_nodes: int = 1,
    devices: str | list[int] | list[str] = "auto",
    precision: str | int = 32,
    load: str | None = None,
    restore_opt: bool = False,
    graph: str = "multiscale",
    hidden_dim: int = 64,
    hidden_layers: int = 1,
    processor_layers: int = 4,
    mesh_aggr: str = "sum",
    output_std: bool = False,
    g2m_gnn_type: str = "InteractionNet",
    m2g_gnn_type: str = "InteractionNet",
    mesh_up_gnn_type: str = "InteractionNet",
    mesh_down_gnn_type: str = "InteractionNet",
    epochs: int = 200,
    batch_size: int = 4,
    ar_steps_train: int = 1,
    loss: str = "wmse",
    lr: float = 1e-3,
    val_interval: int = 1,
    num_sanity_val_steps: int = 2,
    eval: str | None = None,
    ar_steps_eval: int = 10,
    n_example_pred: int = 1,
    create_gif: bool = False,
    logger: str = "wandb",
    logger_project: str = "neural_lam",
    logger_run_name: str | None = None,
    runs_root: str = "runs",
    wandb_id: str | None = None,
    val_steps_to_log: list[int] | None = None,
    train_steps_to_log: list[int] | None = None,
    metrics_watch: list[str] | None = None,
    var_leads_metrics_watch: str = "{}",
    num_past_forcing_steps: int = 1,
    num_future_forcing_steps: int = 1,
    load_single_member: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    **kwargs: Any,
) -> Run:
    """
    Train a Neural-LAM model programmatically.

    Parameters
    ----------
    config_path : str or None, optional
        Path to the Neural-LAM configuration YAML file.
    model : str, default "graph_lam"
        Model architecture name.
    seed : int, default 42
        Random seed for reproducibility.
    num_workers : int, default 4
        Number of workers for data loaders.
    num_nodes : int, default 1
        Number of nodes for DDP distributed training.
    devices : str or list of int or list of str, default "auto"
        Device specification for training.
    precision : str or int, default 32
        Numerical precision (32, 16, or "bf16").
    load : str or None, optional
        Path to checkpoint to resume training from.
    restore_opt : bool, default False
        Whether to restore optimizer state when loading checkpoint.
    graph : str, default "multiscale"
        Name of the graph to load from the datastore graph directory.
    hidden_dim : int, default 64
        Dimensionality of hidden node/edge representations.
    hidden_layers : int, default 1
        Number of hidden layers in MLPs.
    processor_layers : int, default 4
        Number of message-passing processor layers.
    mesh_aggr : str, default "sum"
        Aggregation method ("sum" or "mean").
    output_std : bool, default False
        Whether the model should additionally output predicted std dev.
    g2m_gnn_type : str, default "InteractionNet"
        GNN layer type for grid-to-mesh encoding.
    m2g_gnn_type : str, default "InteractionNet"
        GNN layer type for mesh-to-grid decoding.
    mesh_up_gnn_type : str, default "InteractionNet"
        GNN layer type for upward mesh message passing.
    mesh_down_gnn_type : str, default "InteractionNet"
        GNN layer type for downward mesh message passing.
    epochs : int, default 200
        Maximum training epochs.
    batch_size : int, default 4
        Batch size per GPU.
    ar_steps_train : int, default 1
        Autoregressive rollout steps during training.
    loss : str, default "wmse"
        Loss metric name.
    lr : float, default 1e-3
        Learning rate.
    val_interval : int, default 1
        Epoch interval between validation runs.
    num_sanity_val_steps : int, default 2
        Number of sanity validation steps.
    eval : str or None, optional
        Evaluation split ("val" or "test"). None runs training.
    ar_steps_eval : int, default 10
        Autoregressive rollout steps during evaluation.
    n_example_pred : int, default 1
        Number of qualitative example predictions to plot.
    create_gif : bool, default False
        Whether to generate GIF animations of prediction rollouts.
    logger : str, default "wandb"
        Experiment tracking logger ("wandb" or "mlflow").
    logger_project : str, default "neural_lam"
        Project name for logger.
    logger_run_name : str or None, optional
        Run name for logger.
    runs_root : str, default "runs"
        Directory where outputs and checkpoints are saved.
    wandb_id : str or None, optional
        WandB run ID to resume.
    val_steps_to_log : list of int or None, optional
        Forecast lead steps to log validation loss for.
    train_steps_to_log : list of int or None, optional
        Forecast lead steps to log training loss for.
    metrics_watch : list of str or None, optional
        Metric names to track in summary logs.
    var_leads_metrics_watch : str, default "{}"
        JSON mapping variable IDs to lead steps for metric tracking.
    num_past_forcing_steps : int, default 1
        Number of past forcing timesteps.
    num_future_forcing_steps : int, default 1
        Number of future forcing timesteps.
    load_single_member : bool, default False
        Whether to load only a single ensemble member.
    config : NeuralLAMConfig or None, optional
        Pre-loaded NeuralLAMConfig object.
    datastore : BaseDatastore or None, optional
        Pre-initialized Datastore instance.
    **kwargs : Any
        Additional keyword arguments forwarded to training runtime.

    Returns
    -------
    Run
        Handle containing run output paths and checkpoint locations.
    """
    if val_steps_to_log is None:
        val_steps_to_log = [1, 2, 3, 5, 10]
    if train_steps_to_log is None:
        train_steps_to_log = []
    if metrics_watch is None:
        metrics_watch = []

    devices_val: str | list[int]
    if isinstance(devices, str):
        devices_val = devices
    elif isinstance(devices, list):
        try:
            devices_val = [int(d) for d in devices]
        except ValueError:
            devices_val = "auto"
    else:
        devices_val = "auto"

    model_config = ModelConfig(
        model=kwargs.get("model", model),
        graph=kwargs.get("graph", graph),
        hidden_dim=kwargs.get("hidden_dim", hidden_dim),
        hidden_layers=kwargs.get("hidden_layers", hidden_layers),
        processor_layers=kwargs.get("processor_layers", processor_layers),
        mesh_aggr=kwargs.get("mesh_aggr", mesh_aggr),
        output_std=kwargs.get("output_std", output_std),
        g2m_gnn_type=kwargs.get("g2m_gnn_type", g2m_gnn_type),
        m2g_gnn_type=kwargs.get("m2g_gnn_type", m2g_gnn_type),
        mesh_up_gnn_type=kwargs.get("mesh_up_gnn_type", mesh_up_gnn_type),
        mesh_down_gnn_type=kwargs.get("mesh_down_gnn_type", mesh_down_gnn_type),
    )
    var_leads = (
        {int(k): v for k, v in json.loads(var_leads_metrics_watch).items()}
        if isinstance(var_leads_metrics_watch, str)
        else var_leads_metrics_watch
    )
    train_config = TrainRunConfig(
        epochs=kwargs.get("epochs", epochs),
        batch_size=kwargs.get("batch_size", batch_size),
        ar_steps_train=kwargs.get("ar_steps_train", ar_steps_train),
        ar_steps_eval=kwargs.get("ar_steps_eval", ar_steps_eval),
        loss=kwargs.get("loss", loss),
        lr=kwargs.get("lr", lr),
        val_interval=kwargs.get("val_interval", val_interval),
        num_sanity_val_steps=kwargs.get(
            "num_sanity_val_steps", num_sanity_val_steps
        ),
        val_steps_to_log=val_steps_to_log,
        train_steps_to_log=train_steps_to_log,
        metrics_watch=metrics_watch,
        var_leads_metrics_watch=var_leads,
        load=kwargs.get("load", load),
        restore_opt=kwargs.get("restore_opt", restore_opt),
        eval=kwargs.get("eval", eval),
        n_example_pred=kwargs.get("n_example_pred", n_example_pred),
        create_gif=kwargs.get("create_gif", create_gif),
    )
    data_config = DataConfig(
        num_past_forcing_steps=kwargs.get(
            "num_past_forcing_steps", num_past_forcing_steps
        ),
        num_future_forcing_steps=kwargs.get(
            "num_future_forcing_steps", num_future_forcing_steps
        ),
        num_workers=kwargs.get("num_workers", num_workers),
        load_single_member=kwargs.get("load_single_member", load_single_member),
    )
    compute_config = ComputeConfig(
        seed=kwargs.get("seed", seed),
        num_nodes=kwargs.get("num_nodes", num_nodes),
        devices=devices_val,
        precision=kwargs.get("precision", precision),
    )
    logging_config = LoggingConfig(
        logger=kwargs.get("logger", logger),
        logger_project=kwargs.get("logger_project", logger_project),
        logger_run_name=kwargs.get("logger_run_name", logger_run_name),
        runs_root=kwargs.get("runs_root", runs_root),
        wandb_id=kwargs.get("wandb_id", wandb_id),
    )
    return train_model_script.fit(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        compute_config=compute_config,
        logging_config=logging_config,
        config=config,
        datastore=datastore,
        config_path=config_path,
    )


def evaluate(
    *,
    config_path: str | None = None,
    load: str | None = None,
    eval: str = "test",
    model: str = "graph_lam",
    seed: int = 42,
    num_workers: int = 4,
    num_nodes: int = 1,
    devices: str | list[int] | list[str] = "auto",
    precision: str | int = 32,
    restore_opt: bool = False,
    graph: str = "multiscale",
    hidden_dim: int = 64,
    hidden_layers: int = 1,
    processor_layers: int = 4,
    mesh_aggr: str = "sum",
    output_std: bool = False,
    g2m_gnn_type: str = "InteractionNet",
    m2g_gnn_type: str = "InteractionNet",
    mesh_up_gnn_type: str = "InteractionNet",
    mesh_down_gnn_type: str = "InteractionNet",
    epochs: int = 200,
    batch_size: int = 4,
    ar_steps_train: int = 1,
    loss: str = "wmse",
    lr: float = 1e-3,
    val_interval: int = 1,
    num_sanity_val_steps: int = 2,
    ar_steps_eval: int = 10,
    n_example_pred: int = 1,
    create_gif: bool = False,
    logger: str = "wandb",
    logger_project: str = "neural_lam",
    logger_run_name: str | None = None,
    runs_root: str = "runs",
    wandb_id: str | None = None,
    val_steps_to_log: list[int] | None = None,
    train_steps_to_log: list[int] | None = None,
    metrics_watch: list[str] | None = None,
    var_leads_metrics_watch: str = "{}",
    num_past_forcing_steps: int = 1,
    num_future_forcing_steps: int = 1,
    load_single_member: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    **kwargs: Any,
) -> Run:
    """
    Evaluate a Neural-LAM model checkpoint programmatically.

    Parameters share the meaning and defaults of :func:`train`, with ``eval``
    defaulting to ``"test"``.

    Returns
    -------
    Run
        Handle containing run output paths and evaluation artifacts.
    """
    return train(
        config_path=config_path,
        load=load,
        eval=eval,
        model=model,
        seed=seed,
        num_workers=num_workers,
        num_nodes=num_nodes,
        devices=devices,
        precision=precision,
        restore_opt=restore_opt,
        graph=graph,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        processor_layers=processor_layers,
        mesh_aggr=mesh_aggr,
        output_std=output_std,
        g2m_gnn_type=g2m_gnn_type,
        m2g_gnn_type=m2g_gnn_type,
        mesh_up_gnn_type=mesh_up_gnn_type,
        mesh_down_gnn_type=mesh_down_gnn_type,
        epochs=epochs,
        batch_size=batch_size,
        ar_steps_train=ar_steps_train,
        loss=loss,
        lr=lr,
        val_interval=val_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        ar_steps_eval=ar_steps_eval,
        n_example_pred=n_example_pred,
        create_gif=create_gif,
        logger=logger,
        logger_project=logger_project,
        logger_run_name=logger_run_name,
        runs_root=runs_root,
        wandb_id=wandb_id,
        val_steps_to_log=val_steps_to_log,
        train_steps_to_log=train_steps_to_log,
        metrics_watch=metrics_watch,
        var_leads_metrics_watch=var_leads_metrics_watch,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        load_single_member=load_single_member,
        config=config,
        datastore=datastore,
        **kwargs,
    )


def create_graph(
    *,
    config_path: str | None = None,
    name: str = "multiscale",
    plot: bool = False,
    levels: int | None = None,
    hierarchical: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    **kwargs: Any,
) -> None:
    """
    Generate graph components programmatically.

    Parameters
    ----------
    config_path : str or None, optional
        Path to the Neural-LAM configuration YAML file.
    name : str, default "multiscale"
        Name to save the graph as under ``graph/<name>``.
    plot : bool, default False
        Whether to generate plots of graph connectivity during generation.
    levels : int or None, optional
        Limit multi-scale mesh to given number of levels.
    hierarchical : bool, default False
        Generate hierarchical mesh graph instead of multi-scale graph.
    config : NeuralLAMConfig or None, optional
        Pre-loaded NeuralLAMConfig object.
    datastore : BaseDatastore or None, optional
        Pre-initialized Datastore instance.
    **kwargs : Any
        Additional keyword arguments forwarded to graph creation.
    """
    if datastore is None:
        if config_path is None:
            raise ValueError(
                "Specify config with config_path or provide a "
                "datastore directly"
            )
        _, datastore = load_config_and_datastore(config_path=config_path)

    # Local
    from .datastore.base import BaseRegularGridDatastore

    if not isinstance(datastore, BaseRegularGridDatastore):
        raise TypeError(
            f"Expected BaseRegularGridDatastore, got {type(datastore)}"
        )

    graph_name = kwargs.get("name", name)
    graph_plot = kwargs.get("plot", plot)
    graph_levels = kwargs.get("levels", levels)
    graph_hierarchical = kwargs.get("hierarchical", hierarchical)

    create_graph_script.create_graph_from_datastore(
        datastore=datastore,
        output_root_path=os.path.join(datastore.root_path, "graph", graph_name),
        n_max_levels=graph_levels,
        hierarchical=graph_hierarchical,
        create_plot=graph_plot,
    )


__all__ = [
    "ComputeConfig",
    "DataConfig",
    "LoggingConfig",
    "ModelConfig",
    "Run",
    "TrainRunConfig",
    "create_graph",
    "evaluate",
    "train",
]
