"""Tests for selective GraphLAM checkpoint initialization.

These tests use minimal Lightning-shaped checkpoints with synthetic model
weights. They verify transfer mechanics, not full Lightning serialization or
whether pretrained representations improve forecast skill.
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import GraphLAM, initialize_graph_lam_from_checkpoint
from tests.dummy_datastore import DummyDatastore


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
def graph_datastore():
    """Create a small datastore and flat graph shared by transfer tests."""
    datastore = DummyDatastore(n_grid_points=25)
    create_flat_graph(datastore)
    return datastore


@pytest.fixture(scope="module")
def different_feature_datastores():
    """Create source and target datastores with different state variables."""
    source_datastore = DummyDatastore(n_grid_points=25)
    target_datastore = ReducedStateDummyDatastore(n_grid_points=25)
    create_flat_graph(source_datastore)
    create_flat_graph(target_datastore)
    return source_datastore, target_datastore


def build_graph_lam(datastore, hidden_dim=4):
    """Build the small GraphLAM configuration used by transfer tests."""
    return GraphLAM(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=hidden_dim,
        hidden_layers=1,
        processor_layers=1,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )


def save_graph_lam_checkpoint(model, path):
    """Save a minimal synthetic checkpoint consumed by the transfer utility."""
    state_dict = {
        f"forecaster.predictor.{key}": value.detach().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {"state_dict": state_dict},
        path,
    )


def clone_state(module):
    """Clone a module state for before/after comparisons."""
    return {key: value.clone() for key, value in module.state_dict().items()}


def assert_state_equal(actual, expected):
    """Assert that a module state exactly matches a saved state."""
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert torch.equal(value, expected[key]), key


def test_transfer_supports_different_state_variable_counts(
    different_feature_datastores, tmp_path
):
    """Keep processor transfer working across different feature dimensions."""
    source_datastore, target_datastore = different_feature_datastores
    source = build_graph_lam(source_datastore)
    target = build_graph_lam(target_datastore)

    assert source.grid_input_dim != target.grid_input_dim
    assert source.grid_output_dim != target.grid_output_dim
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
        assert torch.equal(getattr(source, name), getattr(target, name)), name

    grid_embedder_before = clone_state(target.grid_embedder)
    output_map_before = clone_state(target.output_map)

    checkpoint_path = tmp_path / "different_features.ckpt"
    save_graph_lam_checkpoint(source, checkpoint_path)
    initialize_graph_lam_from_checkpoint(
        target,
        checkpoint_path,
        modules=["processor"],
    )

    assert_state_equal(
        target.processor.state_dict(), source.processor.state_dict()
    )
    assert_state_equal(target.grid_embedder.state_dict(), grid_embedder_before)
    assert_state_equal(target.output_map.state_dict(), output_map_before)

    batch_size = 2
    num_grid_nodes = target.num_grid_nodes
    num_state_vars = target_datastore.get_num_data_vars("state")
    num_forcing_vars = target_datastore.get_num_data_vars("forcing")
    prev_state = torch.randn(batch_size, num_grid_nodes, num_state_vars)
    prev_prev_state = torch.randn(batch_size, num_grid_nodes, num_state_vars)
    forcing = torch.randn(batch_size, num_grid_nodes, num_forcing_vars)

    prediction, prediction_std = target(
        prev_state,
        prev_prev_state,
        forcing,
    )
    assert prediction.shape == prev_state.shape
    assert prediction_std is None

    prediction.square().mean().backward()
    processor_gradients = [
        parameter.grad for parameter in target.processor.parameters()
    ]
    assert all(gradient is not None for gradient in processor_gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in processor_gradients
    )


def test_incompatible_processor_shapes_do_not_modify_target(
    graph_datastore, tmp_path
):
    """Reject incompatible shapes before changing any target parameters."""
    source = build_graph_lam(graph_datastore, hidden_dim=4)
    target = build_graph_lam(graph_datastore, hidden_dim=8)
    target_before = clone_state(target.processor)

    checkpoint_path = tmp_path / "incompatible.ckpt"
    save_graph_lam_checkpoint(source, checkpoint_path)

    with pytest.raises(ValueError, match="Incompatible parameter shapes"):
        initialize_graph_lam_from_checkpoint(
            target,
            checkpoint_path,
            modules=["processor"],
        )

    assert_state_equal(target.processor.state_dict(), target_before)


def test_incomplete_processor_state_does_not_modify_target(
    graph_datastore, tmp_path
):
    """Reject a missing processor parameter before changing the target."""
    source = build_graph_lam(graph_datastore)
    target = build_graph_lam(graph_datastore)
    target_before = clone_state(target.processor)

    checkpoint_path = tmp_path / "incomplete.ckpt"
    save_graph_lam_checkpoint(source, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    missing_key = next(
        key
        for key in checkpoint["state_dict"]
        if key.startswith("forecaster.predictor.processor.")
    )
    del checkpoint["state_dict"][missing_key]
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Incompatible state keys"):
        initialize_graph_lam_from_checkpoint(
            target,
            checkpoint_path,
            modules=["processor"],
        )

    assert_state_equal(target.processor.state_dict(), target_before)


@pytest.mark.parametrize(
    ("modules", "match"),
    [
        ([], "At least one GraphLAM module"),
        (["grid_embedder"], "Unsupported GraphLAM transfer modules"),
    ],
)
def test_rejects_invalid_module_selection(
    graph_datastore, tmp_path, modules, match
):
    """Reject empty or unsupported module selections before loading a file."""
    target = build_graph_lam(graph_datastore)

    with pytest.raises(ValueError, match=match):
        initialize_graph_lam_from_checkpoint(
            target,
            tmp_path / "unused.ckpt",
            modules=modules,
        )
