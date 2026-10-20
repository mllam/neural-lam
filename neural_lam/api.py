"""High-level Python API for Neural-LAM."""

# Standard library
import json
import os
from pathlib import Path
from typing import Any, Literal

# Third-party
import numpy as np
import torch
import xarray as xr

# Local
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
from .create_graph_with_wmg import create_graph_from_datastore
from .datastore.base import BaseDatastore
from .train_model import Run
from .utils import create_forecast_dataset
from .weather_dataset import WeatherDataset


def train(
    *,
    config_path: str | None = None,
    model: str = "graph_lam",
    seed: int = 42,
    num_workers: int = 4,
    num_nodes: int = 1,
    devices: int | str | list[int] | list[str] | None = None,
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
    eval: Literal["val", "test"] | None = None,
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
    var_leads_metrics_watch: dict[int, list[int]] | str | None = None,
    num_past_forcing_steps: int = 1,
    num_future_forcing_steps: int = 1,
    num_past_boundary_steps: int = 1,
    num_future_boundary_steps: int = 1,
    load_single_member: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    datastore_boundary: BaseDatastore | None = None,
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
    devices : int or str or list of int or list of str, default "auto"
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
    var_leads_metrics_watch : dict or str or None, optional
        Mapping from variable IDs to lead steps for metric tracking,
        or a JSON string representation.
    num_past_forcing_steps : int, default 1
        Number of past forcing timesteps.
    num_future_forcing_steps : int, default 1
        Number of future forcing timesteps.
    num_past_boundary_steps : int, default 1
        Number of past boundary forcing timesteps.
    num_future_boundary_steps : int, default 1
        Number of future boundary forcing timesteps.
    load_single_member : bool, default False
        Whether to load only a single ensemble member.
    config : NeuralLAMConfig or None, optional
        Pre-loaded NeuralLAMConfig object.
    datastore : BaseDatastore or None, optional
        Pre-initialized Datastore instance.
    datastore_boundary : BaseDatastore or None, optional
        Pre-initialized boundary Datastore instance.

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

    model_config = ModelConfig(
        model=model,
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
    )
    var_leads: dict[int, list[int]]
    if var_leads_metrics_watch is None:
        var_leads = {}
    elif isinstance(var_leads_metrics_watch, str):
        var_leads = {
            int(k): v for k, v in json.loads(var_leads_metrics_watch).items()
        }
    else:
        var_leads = {int(k): v for k, v in var_leads_metrics_watch.items()}
    train_config = TrainRunConfig(
        epochs=epochs,
        batch_size=batch_size,
        ar_steps_train=ar_steps_train,
        ar_steps_eval=ar_steps_eval,
        loss=loss,
        lr=lr,
        val_interval=val_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        val_steps_to_log=val_steps_to_log,
        train_steps_to_log=train_steps_to_log,
        metrics_watch=metrics_watch,
        var_leads_metrics_watch=var_leads,
        load=load,
        restore_opt=restore_opt,
        eval=eval,
        n_example_pred=n_example_pred,
        create_gif=create_gif,
    )
    data_config = DataConfig(
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        num_past_boundary_steps=num_past_boundary_steps,
        num_future_boundary_steps=num_future_boundary_steps,
        num_workers=num_workers,
        load_single_member=load_single_member,
    )
    compute_kwargs: dict[str, Any] = dict(
        seed=seed,
        num_nodes=num_nodes,
        precision=precision,
    )
    if devices is not None:
        compute_kwargs["devices"] = devices
    compute_config = ComputeConfig(**compute_kwargs)
    logging_config = LoggingConfig(
        logger=logger,
        logger_project=logger_project,
        logger_run_name=logger_run_name,
        runs_root=runs_root,
        wandb_id=wandb_id,
    )
    return train_model_script.fit(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        compute_config=compute_config,
        logging_config=logging_config,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        config_path=config_path,
    )


def evaluate(
    *,
    config_path: str | None = None,
    load: str | None = None,
    eval: Literal["val", "test"] = "test",
    model: str = "graph_lam",
    seed: int = 42,
    num_workers: int = 4,
    num_nodes: int = 1,
    devices: int | str | list[int] | list[str] | None = None,
    precision: str | int = 32,
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
    batch_size: int = 4,
    ar_steps_eval: int = 10,
    n_example_pred: int = 1,
    create_gif: bool = False,
    logger: str = "wandb",
    logger_project: str = "neural_lam",
    logger_run_name: str | None = None,
    runs_root: str = "runs",
    wandb_id: str | None = None,
    val_steps_to_log: list[int] | None = None,
    metrics_watch: list[str] | None = None,
    var_leads_metrics_watch: dict[int, list[int]] | str | None = None,
    num_past_forcing_steps: int = 1,
    num_future_forcing_steps: int = 1,
    num_past_boundary_steps: int = 1,
    num_future_boundary_steps: int = 1,
    load_single_member: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    datastore_boundary: BaseDatastore | None = None,
) -> Run:
    """
    Evaluate a Neural-LAM model checkpoint programmatically.

    Parameters share the meaning and defaults of :func:`train`, with ``eval``
    defaulting to ``"test"``, and training-only parameters omitted.

    Returns
    -------
    Run
        Handle containing run output paths and evaluation artifacts.
    """
    if val_steps_to_log is None:
        val_steps_to_log = [1, 2, 3, 5, 10]
    if metrics_watch is None:
        metrics_watch = []

    model_config = ModelConfig(
        model=model,
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
    )
    var_leads: dict[int, list[int]]
    if var_leads_metrics_watch is None:
        var_leads = {}
    elif isinstance(var_leads_metrics_watch, str):
        var_leads = {
            int(k): v for k, v in json.loads(var_leads_metrics_watch).items()
        }
    else:
        var_leads = {int(k): v for k, v in var_leads_metrics_watch.items()}
    train_config = TrainRunConfig(
        epochs=1,
        batch_size=batch_size,
        ar_steps_train=1,
        ar_steps_eval=ar_steps_eval,
        loss="wmse",
        lr=1e-3,
        val_interval=1,
        num_sanity_val_steps=0,
        val_steps_to_log=val_steps_to_log,
        train_steps_to_log=[],
        metrics_watch=metrics_watch,
        var_leads_metrics_watch=var_leads,
        load=load,
        restore_opt=False,
        eval=eval,
        n_example_pred=n_example_pred,
        create_gif=create_gif,
    )
    data_config = DataConfig(
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        num_past_boundary_steps=num_past_boundary_steps,
        num_future_boundary_steps=num_future_boundary_steps,
        num_workers=num_workers,
        load_single_member=load_single_member,
    )
    compute_kwargs: dict[str, Any] = dict(
        seed=seed,
        num_nodes=num_nodes,
        precision=precision,
    )
    if devices is not None:
        compute_kwargs["devices"] = devices
    compute_config = ComputeConfig(**compute_kwargs)
    logging_config = LoggingConfig(
        logger=logger,
        logger_project=logger_project,
        logger_run_name=logger_run_name,
        runs_root=runs_root,
        wandb_id=wandb_id,
    )
    return train_model_script.fit(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        compute_config=compute_config,
        logging_config=logging_config,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        config_path=config_path,
    )


def create_graph(
    *,
    config_path: str | None = None,
    datastore: BaseDatastore | None = None,
    name: str = "multiscale",
    output_root_path: str | Path | None = None,
    archetype: Literal["keisler", "graphcast", "hierarchical"] = "keisler",
    mesh_node_distance: float | None = None,
    mesh_grid_distance_ratio: float = 3.0,
    level_refinement_factor: int = 3,
    max_num_levels: int | None = None,
) -> None:
    """
    Generate graph components programmatically using weather-model-graphs.

    Parameters
    ----------
    config_path : str or None, optional
        Path to the Neural-LAM configuration YAML file. Used to load the
        datastore if ``datastore`` is not provided.
    datastore : BaseDatastore or None, optional
        Datastore instance providing grid coordinates.
    name : str, default "multiscale"
        Name of the graph subdirectory under ``graph/<name>``.
    output_root_path : str or Path or None, optional
        Directory where graph files will be saved. Defaults to
        ``<datastore.root_path>/graph/<name>``.
    archetype : {"keisler", "graphcast", "hierarchical"}, default "keisler"
        Graph archetype to create.
    mesh_node_distance : float or None, optional
        Distance between mesh nodes in coordinate units.
    mesh_grid_distance_ratio : float, default 3.0
        Ratio of mesh node distance to grid node distance.
    level_refinement_factor : int, default 3
        Refinement factor between mesh levels.
    max_num_levels : int or None, optional
        Maximum number of mesh levels.
    """
    if datastore is None:
        if config_path is None:
            raise ValueError(
                "Either datastore or config_path must be provided."
            )
        _, datastore, _ = load_config_and_datastore(config_path=config_path)

    if output_root_path is None:
        output_root_path = os.path.join(datastore.root_path, "graph", name)

    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=str(output_root_path),
        archetype=archetype,
        mesh_node_distance=mesh_node_distance,
        mesh_grid_distance_ratio=mesh_grid_distance_ratio,
        level_refinement_factor=level_refinement_factor,
        max_num_levels=max_num_levels,
    )


def predict(
    *,
    load: str,
    config_path: str | None = None,
    init_times: (
        str
        | np.datetime64
        | list[str | np.datetime64]
        | slice
        | int
        | list[int]
        | None
    ) = None,
    lead_steps: int | None = None,
    split: str = "test",
    device: str | torch.device = "cpu",
    batch_size: int = 1,
    unstandardize: bool = True,
    return_target: bool = False,
    unstack_grid: bool = False,
    config: NeuralLAMConfig | None = None,
    datastore: BaseDatastore | None = None,
    datastore_boundary: BaseDatastore | None = None,
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset]:
    """
    Generate weather forecast predictions using a trained model checkpoint.

    Parameters
    ----------
    load : str
        Path to the saved model checkpoint (.ckpt).
    config_path : str or None, optional
        Path to the Neural-LAM YAML configuration. Required if config or
        datastore is not provided.
    init_times : str, np.datetime64, list, slice, int, or None, optional
        Forecast initialization time(s). Can be a timestamp string
        (e.g. ``'2022-04-01T00:00:00'``), list of timestamps, timestamp slice,
        integer sample index, or list of sample indices. If ``None`` (default),
        uses the first available initialization time in the split.
    lead_steps : int or None, optional
        Number of autoregressive forecast steps to rollout. Defaults to the
        configured evaluation steps or 10.
    split : str, default "test"
        Data split to extract initial conditions and forcing from
        (``'train'``, ``'val'``, or ``'test'``).
    device : str or torch.device, default "cpu"
        Compute device to perform inference on (e.g. ``'cpu'`` or ``'cuda'``).
    batch_size : int, default 1
        Batch size for inference rollouts.
    unstandardize : bool, default True
        Whether to unstandardize predictions back into physical units using the
        dataset state mean and standard deviation.
    return_target : bool, default False
        Whether to also return the ground-truth target states as a companion
        dataset.
    unstack_grid : bool, default False
        Whether to unstack the ``grid_index`` dimension into spatial grid
        dimensions (e.g. ``x``, ``y``) for regular grid datastores.
    config : NeuralLAMConfig or None, optional
        Loaded Neural-LAM configuration dataclass.
    datastore : BaseDatastore or None, optional
        Initialized datastore instance.
    datastore_boundary : BaseDatastore or None, optional
        Initialized boundary datastore instance.

    Returns
    -------
    xr.Dataset or tuple of (xr.Dataset, xr.Dataset)
        Forecast prediction dataset, or tuple of ``(predictions, targets)``
        when ``return_target=True``.
    """
    if config is None or datastore is None:
        if config_path is None:
            raise ValueError(
                "Either (config and datastore) or config_path must be provided."
            )
        config, datastore, datastore_boundary = load_config_and_datastore(
            config_path=config_path
        )

    model = train_model_script.load_forecaster_module_from_checkpoint(
        ckpt_path=load,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
    )
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    if lead_steps is None:
        training_cfg = getattr(config, "training", None)
        ar_steps_eval = getattr(training_cfg, "ar_steps_eval", None)
        lead_steps = ar_steps_eval if ar_steps_eval is not None else 10

    args = getattr(model.hparams, "args", None)
    num_past = getattr(args, "num_past_forcing_steps", 1) if args else 1
    num_future = getattr(args, "num_future_forcing_steps", 1) if args else 1
    num_past_boundary = (
        getattr(args, "num_past_boundary_steps", 1) if args else 1
    )
    num_future_boundary = (
        getattr(args, "num_future_boundary_steps", 1) if args else 1
    )
    data_cfg = getattr(config, "data", None)
    if data_cfg is not None:
        num_past = getattr(data_cfg, "num_past_forcing_steps", num_past)
        num_future = getattr(data_cfg, "num_future_forcing_steps", num_future)
        num_past_boundary = getattr(
            data_cfg, "num_past_boundary_steps", num_past_boundary
        )
        num_future_boundary = getattr(
            data_cfg, "num_future_boundary_steps", num_future_boundary
        )

    dataset = WeatherDataset(
        datastore=datastore,
        split=split,
        ar_steps=lead_steps,
        num_past_forcing_steps=num_past,
        num_future_forcing_steps=num_future,
        num_past_boundary_steps=num_past_boundary,
        num_future_boundary_steps=num_future_boundary,
        datastore_boundary=datastore_boundary,
    )

    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError(
            f"Dataset for split '{split}' with lead_steps={lead_steps} "
            "contains 0 samples. Not enough time steps available."
        )

    n_members = (
        dataset.da_state.ensemble_member.size
        if hasattr(dataset.da_state, "ensemble_member")
        else 1
    )
    base_init_times, _, _ = dataset._sample_window_times()
    base_init_times = np.asarray(base_init_times, dtype="datetime64[ns]")

    if datastore.is_ensemble and not dataset.load_single_member:
        all_init_times = np.repeat(base_init_times, n_members)
    else:
        all_init_times = base_init_times

    if init_times is None:
        selected_indices = [0]
    elif isinstance(init_times, (int, np.integer)):
        idx = int(init_times)
        if not 0 <= idx < n_samples:
            raise IndexError(
                f"Sample index {idx} out of range [0, {n_samples - 1}]."
            )
        selected_indices = [idx]
    elif isinstance(init_times, slice):
        start_val = init_times.start
        stop_val = init_times.stop
        step_val = init_times.step or 1

        if (start_val is None or isinstance(start_val, (int, np.integer))) and (
            stop_val is None or isinstance(stop_val, (int, np.integer))
        ):
            selected_indices = list(range(n_samples)[init_times])
        else:
            start_dt = (
                np.asarray(start_val, dtype="datetime64[ns]")
                if start_val is not None
                else None
            )
            stop_dt = (
                np.asarray(stop_val, dtype="datetime64[ns]")
                if stop_val is not None
                else None
            )
            mask = np.ones(n_samples, dtype=bool)
            if start_dt is not None:
                mask &= all_init_times >= start_dt
            if stop_dt is not None:
                mask &= all_init_times <= stop_dt
            selected_indices = np.where(mask)[0][::step_val].tolist()
            if len(selected_indices) == 0:
                raise ValueError(
                    f"No samples match init_times slice '{init_times}'. "
                    f"Available times range from {all_init_times[0]} to "
                    f"{all_init_times[-1]}."
                )
    elif isinstance(init_times, (str, np.datetime64)):
        target_dt = np.asarray(init_times, dtype="datetime64[ns]")
        matches = np.where(all_init_times == target_dt)[0]
        if len(matches) == 0:
            raise ValueError(
                f"Requested init_time '{init_times}' not found. "
                f"Available init_times range from {all_init_times[0]} to "
                f"{all_init_times[-1]}."
            )
        selected_indices = [int(matches[0])]
    elif hasattr(init_times, "__iter__"):
        selected_indices = []
        for t in init_times:
            if isinstance(t, (int, np.integer)):
                idx = int(t)
                if not 0 <= idx < n_samples:
                    raise IndexError(
                        f"Sample index {idx} out of range [0, {n_samples - 1}]."
                    )
                selected_indices.append(idx)
            else:
                target_dt = np.asarray(t, dtype="datetime64[ns]")
                matches = np.where(all_init_times == target_dt)[0]
                if len(matches) == 0:
                    raise ValueError(
                        f"Requested init_time '{t}' not found. "
                        f"Available init_times range from {all_init_times[0]} "
                        f"to {all_init_times[-1]}."
                    )
                selected_indices.append(int(matches[0]))
    else:
        raise TypeError(f"Unsupported type for init_times: {type(init_times)}")

    pred_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    pred_std_list: list[torch.Tensor] = []

    for batch_start in range(0, len(selected_indices), batch_size):
        batch_idxs = selected_indices[batch_start : batch_start + batch_size]
        batch_samples = [dataset[i] for i in batch_idxs]

        init_batch = torch.stack([s[0] for s in batch_samples], dim=0).to(
            torch_device
        )
        target_batch = torch.stack([s[1] for s in batch_samples], dim=0).to(
            torch_device
        )
        forcing_batch = torch.stack([s[2] for s in batch_samples], dim=0).to(
            torch_device
        )
        boundary_batch = torch.stack([s[3] for s in batch_samples], dim=0).to(
            torch_device
        )
        times_batch = torch.stack([s[4] for s in batch_samples], dim=0).to(
            torch_device
        )

        (
            norm_init,
            norm_target,
            norm_forcing,
            norm_boundary,
            _,
        ) = model.on_after_batch_transfer(
            (
                init_batch,
                target_batch,
                forcing_batch,
                boundary_batch,
                times_batch,
            ),
            dataloader_idx=0,
        )

        with torch.no_grad():
            pred, pred_std = model.forecaster(
                norm_init, norm_forcing, norm_target
            )

        if unstandardize:
            pred = pred * model.state_std + model.state_mean
            target_to_save = target_batch
            if pred_std is not None:
                pred_std = pred_std * model.state_std
        else:
            target_to_save = norm_target

        pred_list.append(pred.cpu())
        target_list.append(target_to_save.cpu())
        if pred_std is not None:
            pred_std_list.append(pred_std.cpu())

    all_preds = torch.cat(pred_list, dim=0)
    all_targets = torch.cat(target_list, dim=0)
    all_pred_stds = torch.cat(pred_std_list, dim=0) if pred_std_list else None

    sel_analysis_times = all_init_times[selected_indices]
    first_target_times = (
        dataset[selected_indices[0]][4].cpu().numpy().astype("datetime64[ns]")
    )
    first_init_time = sel_analysis_times[0]
    elapsed_durations = first_target_times - first_init_time

    ds_pred = create_forecast_dataset(
        predictions=all_preds,
        analysis_times=sel_analysis_times,
        elapsed_durations=elapsed_durations,
        datastore=datastore,
        category="state",
        pred_std=all_pred_stds,
    )

    if unstack_grid:
        ds_pred = datastore.unstack_grid_coords(ds_pred)

    if return_target:
        ds_target = create_forecast_dataset(
            predictions=all_targets,
            analysis_times=sel_analysis_times,
            elapsed_durations=elapsed_durations,
            datastore=datastore,
            category="state",
        )
        if unstack_grid:
            ds_target = datastore.unstack_grid_coords(ds_target)
        return ds_pred, ds_target

    return ds_pred


__all__ = [
    "ComputeConfig",
    "DataConfig",
    "LoggingConfig",
    "ModelConfig",
    "Run",
    "TrainRunConfig",
    "create_forecast_dataset",
    "create_graph",
    "evaluate",
    "predict",
    "train",
]
