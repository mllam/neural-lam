"""Tests for selective component initialization from Lightning checkpoints."""

# Standard library
from argparse import Namespace
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import (
    DeterministicARForecaster,
    DeterministicForecastingModule,
    GraphLAM,
)
from neural_lam.train_model import _checkpoint_loading_kwargs
from tests.dummy_datastore import DummyDatastore

TRANSFER_COMPONENTS = (
    "forecaster.predictor.processor",
    "forecaster.predictor.mesh_embedder",
    "forecaster.predictor.m2m_embedder",
)


class ReducedStateDummyDatastore(DummyDatastore):
    """Dummy datastore with fewer target state variables."""

    N_FEATURES = dict(DummyDatastore.N_FEATURES, state=3)


def create_flat_graph(datastore):
    """Create the one-level graph required by the test GraphLAM models."""
    graph_dir = Path(datastore.root_path) / "graph" / "1level"
    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=str(graph_dir),
        n_max_levels=1,
    )


@pytest.fixture(scope="module")
def different_feature_datastores():
    """Create source and target datastores with different state variables."""
    source_datastore = DummyDatastore(n_grid_points=81)
    target_datastore = ReducedStateDummyDatastore(n_grid_points=81)
    create_flat_graph(source_datastore)
    create_flat_graph(target_datastore)
    return source_datastore, target_datastore


def build_module(
    datastore,
    hidden_dim=4,
    *,
    loss="mse",
    lr=1.0e-3,
    n_example_pred=1,
    create_gif=False,
    val_steps_to_log=None,
    train_steps_to_log=None,
    metrics_watch=None,
    restore_opt=False,
    args=None,
):
    """Build a small deterministic GraphLAM forecasting module."""
    if val_steps_to_log is None:
        val_steps_to_log = [1]
    if train_steps_to_log is None:
        train_steps_to_log = []
    if metrics_watch is None:
        metrics_watch = []
    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )
    predictor = GraphLAM(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=hidden_dim,
        hidden_layers=1,
        processor_layers=1,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    forecaster = DeterministicARForecaster(
        predictor,
        datastore,
        config=config,
        loss=loss,
    )
    module = DeterministicForecastingModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        lr=lr,
        restore_opt=restore_opt,
        n_example_pred=n_example_pred,
        create_gif=create_gif,
        val_steps_to_log=val_steps_to_log,
        train_steps_to_log=train_steps_to_log,
        metrics_watch=metrics_watch,
        args=args,
    )
    return module, config


def save_lightning_checkpoint(module, path):
    """Save a real Lightning checkpoint without running a training epoch."""
    trainer = pl.Trainer(
        default_root_dir=path.parent,
        accelerator="cpu",
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path, weights_only=False)


def clone_state(module):
    """Clone a module state for before/after comparisons."""
    return {key: value.clone() for key, value in module.state_dict().items()}


def assert_state_equal(actual, expected):
    """Assert that a module state exactly matches a saved state."""
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert torch.equal(value, expected[key]), key


def load_selected_components(
    checkpoint_path,
    target_module,
    target_config,
    target_datastore,
    components=TRANSFER_COMPONENTS,
    allow_missing_compatibility_metadata=False,
):
    """Load selected source components into a fresh target module."""
    return DeterministicForecastingModule.load_from_checkpoint(
        checkpoint_path,
        forecaster=target_module.forecaster,
        config=target_config,
        datastore=target_datastore,
        datastore_boundary=target_module.datastore_boundary,
        lr=target_module.hparams.lr,
        restore_opt=target_module.restore_opt,
        n_example_pred=target_module.n_example_pred,
        create_gif=target_module.create_gif,
        val_steps_to_log=target_module.hparams.val_steps_to_log,
        train_steps_to_log=target_module.hparams.train_steps_to_log,
        metrics_watch=target_module.hparams.metrics_watch,
        var_leads_metrics_watch=target_module.hparams.var_leads_metrics_watch,
        args=target_module.hparams.args,
        pretrained_components=list(components),
        allow_missing_compatibility_metadata=(
            allow_missing_compatibility_metadata
        ),
        map_location="cpu",
        strict=True,
        **_checkpoint_loading_kwargs(),
    )


def test_selective_loading_across_state_variable_counts(
    different_feature_datastores, tmp_path
):
    """Load compatible GraphLAM components across dataset dimensions."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    source_predictor = source.forecaster.predictor
    target_predictor = target.forecaster.predictor

    assert source_predictor.grid_input_dim != target_predictor.grid_input_dim
    assert source_predictor.grid_output_dim != target_predictor.grid_output_dim
    graph_tensors = (
        "m2m_edge_index",
        "m2m_features",
        "mesh_static_features",
        "g2m_edge_index",
        "m2g_edge_index",
        "g2m_features",
        "m2g_features",
    )
    for name in graph_tensors:
        assert torch.equal(
            getattr(source_predictor, name),
            getattr(target_predictor, name),
        ), name

    with torch.no_grad():
        for index, path in enumerate(TRANSFER_COMPONENTS, start=1):
            for parameter in source.get_submodule(path).parameters():
                parameter.fill_(index / 10)

    target_state_before = clone_state(target)
    checkpoint_path = tmp_path / "source.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)
    source_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["state_dict"]

    selected_prefixes = tuple(f"{path}." for path in TRANSFER_COMPONENTS)
    assert any(
        not torch.equal(target_state_before[key], source_state[key])
        for key in target_state_before
        if key.startswith(selected_prefixes)
    )

    loaded = load_selected_components(
        checkpoint_path,
        target,
        target_config,
        target_datastore,
    )
    loaded_state = loaded.state_dict()
    for key, value in loaded_state.items():
        if key.startswith(selected_prefixes):
            assert torch.equal(value, source_state[key]), key
        else:
            assert torch.equal(value, target_state_before[key]), key

    loaded.freeze_components(["forecaster.predictor.processor"])
    processor = loaded.forecaster.predictor.processor
    assert all(
        not parameter.requires_grad for parameter in processor.parameters()
    )

    predictor = loaded.forecaster.predictor
    batch_size = 2
    num_grid_nodes = predictor.num_grid_nodes
    num_state_vars = target_datastore.get_num_data_vars("state")
    num_forcing_vars = target_datastore.get_num_data_vars("forcing")
    prev_state = torch.randn(batch_size, num_grid_nodes, num_state_vars)
    prev_prev_state = torch.randn(batch_size, num_grid_nodes, num_state_vars)
    forcing = torch.randn(batch_size, num_grid_nodes, num_forcing_vars)

    prediction, prediction_std = predictor(
        prev_state,
        prev_prev_state,
        forcing,
    )
    assert prediction.shape == prev_state.shape
    assert prediction_std is None

    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in processor.parameters())
    for path in TRANSFER_COMPONENTS[1:]:
        trainable_gradients = [
            parameter.grad
            for parameter in loaded.get_submodule(path).parameters()
        ]
        assert all(gradient is not None for gradient in trainable_gradients)
        assert all(
            torch.isfinite(gradient).all() for gradient in trainable_gradients
        )


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_invalid_component_state_does_not_modify_target(
    different_feature_datastores, tmp_path, failure
):
    """Reject an incomplete or shape-incompatible component atomically."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    target_state_before = clone_state(target)
    checkpoint_path = tmp_path / f"invalid_{failure}.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    processor_key = next(
        key
        for key in checkpoint["state_dict"]
        if key.startswith("forecaster.predictor.processor.")
    )
    if failure == "missing":
        del checkpoint["state_dict"][processor_key]
        error = "Incompatible state keys"
    else:
        checkpoint["state_dict"][processor_key] = torch.empty(1)
        error = "Incompatible parameter shapes"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=error):
        load_selected_components(
            checkpoint_path,
            target,
            target_config,
            target_datastore,
            components=("forecaster.predictor.processor",),
            allow_missing_compatibility_metadata=(failure == "mismatch"),
        )

    assert_state_equal(target.state_dict(), target_state_before)


def test_legacy_checkpoint_requires_explicit_compatibility_override(
    different_feature_datastores, tmp_path
):
    """Warn and load legacy weights only when the caller opts in."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    checkpoint_path = tmp_path / "legacy_without_metadata.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    del checkpoint["neural_lam_pretrained_compatibility"]
    torch.save(checkpoint, checkpoint_path)

    with pytest.warns(
        UserWarning,
        match="caller is responsible for graph compatibility",
    ):
        loaded = load_selected_components(
            checkpoint_path,
            target,
            target_config,
            target_datastore,
            components=("forecaster.predictor.processor",),
            allow_missing_compatibility_metadata=True,
        )

    assert_state_equal(
        loaded.forecaster.predictor.processor.state_dict(),
        source.forecaster.predictor.processor.state_dict(),
    )


def test_multiple_component_validation_is_atomic(
    different_feature_datastores, tmp_path
):
    """Do not load an earlier valid component if a later one is invalid."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    target_state_before = clone_state(target)
    checkpoint_path = tmp_path / "multiple_components.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    with pytest.raises(ValueError, match="Incompatible parameter shapes"):
        load_selected_components(
            checkpoint_path,
            target,
            target_config,
            target_datastore,
            components=(
                "forecaster.predictor.processor",
                "forecaster.predictor.output_map",
            ),
        )

    assert_state_equal(target.state_dict(), target_state_before)


@pytest.mark.parametrize("failure", ["missing", "mismatch"])
def test_graph_compatibility_is_required(
    different_feature_datastores, tmp_path, failure
):
    """Reject missing or mismatched graph compatibility metadata."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    target_state_before = clone_state(target)
    checkpoint_path = tmp_path / f"graph_{failure}.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    if failure == "missing":
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        del checkpoint["neural_lam_pretrained_compatibility"]
        torch.save(checkpoint, checkpoint_path)
        error = "does not contain compatibility metadata"
    else:
        target.forecaster.predictor.m2m_edge_index[0, 0] += 1
        error = "incompatible with the target model graph"

    with pytest.raises(ValueError, match=error):
        load_selected_components(
            checkpoint_path,
            target,
            target_config,
            target_datastore,
            components=("forecaster.predictor.processor",),
        )

    assert_state_equal(target.state_dict(), target_state_before)


def test_target_hyperparameters_override_source_checkpoint(
    different_feature_datastores, tmp_path
):
    """Keep target forecasting and training settings during initialization."""
    source_datastore, target_datastore = different_feature_datastores
    source_args = Namespace(
        lr=0.1,
        restore_opt=False,
        n_example_pred=7,
        create_gif=True,
        val_steps_to_log=[2],
        train_steps_to_log=[2],
        metrics_watch=["source_metric"],
        var_leads_metrics_watch={},
    )
    target_args = Namespace(
        lr=0.002,
        restore_opt=False,
        n_example_pred=3,
        create_gif=False,
        val_steps_to_log=[1],
        train_steps_to_log=[1],
        metrics_watch=["target_metric"],
        var_leads_metrics_watch={},
    )
    source, _ = build_module(
        source_datastore,
        loss="mse",
        lr=0.1,
        n_example_pred=7,
        create_gif=True,
        val_steps_to_log=[2],
        train_steps_to_log=[2],
        metrics_watch=["source_metric"],
        args=source_args,
    )
    target, target_config = build_module(
        target_datastore,
        loss="wmse",
        lr=0.002,
        n_example_pred=3,
        create_gif=False,
        val_steps_to_log=[1],
        train_steps_to_log=[1],
        metrics_watch=["target_metric"],
        args=target_args,
    )
    checkpoint_path = tmp_path / "source_hparams.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    loaded = load_selected_components(
        checkpoint_path,
        target,
        target_config,
        target_datastore,
        components=("forecaster.predictor.processor",),
    )

    assert loaded.hparams.lr == target.hparams.lr
    assert loaded.n_example_pred == target.n_example_pred
    assert loaded.create_gif == target.create_gif
    assert loaded.hparams.val_steps_to_log == [1]
    assert loaded.hparams.train_steps_to_log == [1]
    assert loaded.hparams.metrics_watch == ["target_metric"]
    assert loaded.hparams.config == target_config
    assert vars(loaded.hparams.args) == vars(target_args)
    assert loaded.forecaster.loss is target.forecaster.loss


def test_transferred_checkpoint_full_reload_does_not_reinitialize(
    different_feature_datastores, tmp_path
):
    """Fully reload transferred state without repeating initialization."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target_args = Namespace(
        lr=0.002,
        restore_opt=False,
        n_example_pred=1,
        create_gif=False,
        val_steps_to_log=[1],
        train_steps_to_log=[],
        metrics_watch=[],
        var_leads_metrics_watch={},
        pretrained_checkpoint="source.ckpt",
        pretrained_components=["forecaster.predictor.processor"],
    )
    target, target_config = build_module(
        target_datastore,
        lr=0.002,
        args=target_args,
    )
    source_checkpoint = tmp_path / "source_resume.ckpt"
    save_lightning_checkpoint(source, source_checkpoint)
    transferred = load_selected_components(
        source_checkpoint,
        target,
        target_config,
        target_datastore,
        components=("forecaster.predictor.processor",),
    )

    transferred_checkpoint = tmp_path / "transferred.ckpt"
    save_lightning_checkpoint(transferred, transferred_checkpoint)
    fresh_target, fresh_config = build_module(target_datastore, lr=0.002)
    resumed = DeterministicForecastingModule.load_from_checkpoint(
        transferred_checkpoint,
        forecaster=fresh_target.forecaster,
        config=fresh_config,
        datastore=target_datastore,
        map_location="cpu",
        strict=True,
        **_checkpoint_loading_kwargs(),
    )

    assert resumed._pretrained_components == ()
    assert_state_equal(resumed.state_dict(), transferred.state_dict())


@pytest.mark.requires_training
def test_transferred_checkpoint_resumes_training_state(
    different_feature_datastores, tmp_path
):
    """Resume optimizer and loop state after a transferred model is saved."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    source_checkpoint = tmp_path / "source_training.ckpt"
    save_lightning_checkpoint(source, source_checkpoint)
    transferred = load_selected_components(
        source_checkpoint,
        target,
        target_config,
        target_datastore,
        components=("forecaster.predictor.processor",),
    )

    batch_size = 1
    pred_steps = 1
    num_grid_nodes = target_datastore.num_grid_points
    num_state_vars = target_datastore.get_num_data_vars("state")
    num_forcing_vars = target_datastore.get_num_data_vars("forcing")
    batch = (
        torch.randn(batch_size, 2, num_grid_nodes, num_state_vars),
        torch.randn(batch_size, pred_steps, num_grid_nodes, num_state_vars),
        torch.randn(batch_size, pred_steps, num_grid_nodes, num_forcing_vars),
        torch.empty(batch_size, pred_steps, num_grid_nodes, 0),
        torch.zeros(batch_size, pred_steps, dtype=torch.int64),
    )
    train_loader = DataLoader([batch], batch_size=None)
    first_trainer = pl.Trainer(
        default_root_dir=tmp_path / "first_run",
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    first_trainer.fit(transferred, train_dataloaders=train_loader)
    assert first_trainer.global_step == 1

    transferred_checkpoint = tmp_path / "trained_transfer.ckpt"
    first_trainer.save_checkpoint(transferred_checkpoint)
    checkpoint = torch.load(transferred_checkpoint, weights_only=False)
    assert checkpoint["optimizer_states"][0]["state"]

    resumed, _ = build_module(target_datastore, restore_opt=True)
    resume_trainer = pl.Trainer(
        default_root_dir=tmp_path / "resumed_run",
        accelerator="cpu",
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    resume_trainer.fit(
        resumed,
        train_dataloaders=train_loader,
        ckpt_path=transferred_checkpoint,
    )
    assert resume_trainer.global_step == 2
    resumed_optimizer_state = resume_trainer.optimizers[0].state_dict()["state"]
    optimizer_steps = [
        int(state["step"].item()) for state in resumed_optimizer_state.values()
    ]
    assert optimizer_steps
    assert min(optimizer_steps) == 2


def test_selective_loading_reuses_legacy_namespace_migration(
    different_feature_datastores, tmp_path
):
    """Apply existing flat-key migration before selecting a component."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    checkpoint_path = tmp_path / "legacy.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    processor_prefix = "forecaster.predictor.processor."
    processor_state = {
        key.removeprefix("forecaster.predictor."): value
        for key, value in list(checkpoint["state_dict"].items())
        if key.startswith(processor_prefix)
    }
    for key in list(checkpoint["state_dict"]):
        if key.startswith(processor_prefix):
            del checkpoint["state_dict"][key]
    checkpoint["state_dict"].update(processor_state)
    torch.save(checkpoint, checkpoint_path)

    loaded = load_selected_components(
        checkpoint_path,
        target,
        target_config,
        target_datastore,
        components=("forecaster.predictor.processor",),
    )
    assert_state_equal(
        loaded.forecaster.predictor.processor.state_dict(),
        source.forecaster.predictor.processor.state_dict(),
    )


@pytest.mark.parametrize(
    ("components", "match"),
    [
        (["forecaster.predictor.unknown"], "Unknown pretrained component"),
        (["forecaster.predictor.grid_embedder.1"], "has no state to load"),
    ],
)
def test_invalid_component_paths_do_not_modify_target(
    different_feature_datastores,
    tmp_path,
    components,
    match,
):
    """Reject invalid component paths without modifying the target."""
    source_datastore, target_datastore = different_feature_datastores
    source, _ = build_module(source_datastore)
    target, target_config = build_module(target_datastore)
    target_state_before = clone_state(target)
    checkpoint_path = tmp_path / "source.ckpt"
    save_lightning_checkpoint(source, checkpoint_path)

    with pytest.raises(ValueError, match=match):
        load_selected_components(
            checkpoint_path,
            target,
            target_config,
            target_datastore,
            components=components,
        )

    assert_state_equal(target.state_dict(), target_state_before)


@pytest.mark.parametrize(
    ("components", "match"),
    [
        (
            [
                "forecaster.predictor.processor",
                "forecaster.predictor.processor",
            ],
            "Duplicate component paths",
        ),
        (
            ["forecaster.predictor", "forecaster.predictor.processor"],
            "must not overlap",
        ),
        (["forecaster..predictor"], "Invalid component paths"),
        ([123], "Component paths must be strings"),
    ],
)
def test_component_path_validation(components, match):
    """Validate component path syntax without constructing a checkpoint."""
    with pytest.raises(ValueError, match=match):
        DeterministicForecastingModule._validate_component_paths(components)
