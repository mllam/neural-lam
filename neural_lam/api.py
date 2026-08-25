"""High-level Python API for Neural-LAM."""

# Standard library
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Local
from . import create_graph as create_graph_script
from . import train_model as train_model_script
from .config import NeuralLAMConfig
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

    devices_list: list[str]
    if isinstance(devices, str):
        devices_list = [devices]
    elif isinstance(devices, list):
        devices_list = [str(d) for d in devices]
    else:
        devices_list = ["auto"]

    args_dict = {
        "config_path": config_path,
        "model": model,
        "seed": seed,
        "num_workers": num_workers,
        "num_nodes": num_nodes,
        "devices": devices_list,
        "precision": str(precision),
        "load": load,
        "restore_opt": restore_opt,
        "graph": graph,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "processor_layers": processor_layers,
        "mesh_aggr": mesh_aggr,
        "output_std": output_std,
        "g2m_gnn_type": g2m_gnn_type,
        "m2g_gnn_type": m2g_gnn_type,
        "mesh_up_gnn_type": mesh_up_gnn_type,
        "mesh_down_gnn_type": mesh_down_gnn_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "ar_steps_train": ar_steps_train,
        "loss": loss,
        "lr": lr,
        "val_interval": val_interval,
        "num_sanity_val_steps": num_sanity_val_steps,
        "eval": eval,
        "ar_steps_eval": ar_steps_eval,
        "n_example_pred": n_example_pred,
        "create_gif": create_gif,
        "logger": logger,
        "logger_project": logger_project,
        "logger_run_name": logger_run_name,
        "runs_root": runs_root,
        "wandb_id": wandb_id,
        "val_steps_to_log": val_steps_to_log,
        "train_steps_to_log": train_steps_to_log,
        "metrics_watch": metrics_watch,
        "var_leads_metrics_watch": var_leads_metrics_watch,
        "num_past_forcing_steps": num_past_forcing_steps,
        "num_future_forcing_steps": num_future_forcing_steps,
        "load_single_member": load_single_member,
    }
    args_dict.update(kwargs)

    args = argparse.Namespace(**args_dict)
    return train_model_script.run(args, config=config, datastore=datastore)


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
    args_dict = {
        "config_path": config_path,
        "name": name,
        "plot": plot,
        "levels": levels,
        "hierarchical": hierarchical,
    }
    args_dict.update(kwargs)
    args = argparse.Namespace(**args_dict)
    create_graph_script.run(args, config=config, datastore=datastore)
