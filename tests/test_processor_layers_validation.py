"""Regression tests for ``processor_layers`` validation in hierarchical models."""

# Standard library
from pathlib import Path

# Third-party
import pytest

# First-party
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.models import HiLAM, HiLAMParallel
from tests.dummy_datastore import DummyDatastore

HIERARCHICAL_MODELS = (HiLAM, HiLAMParallel)


@pytest.fixture(scope="module")
def hierarchical_datastore():
    """Build a small hierarchical mesh graph on an in-memory datastore."""
    datastore = DummyDatastore(n_grid_points=400, n_timesteps=3)
    graph_dir_path = Path(datastore.root_path) / "graph" / "hierarchical"
    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=str(graph_dir_path),
        archetype="hierarchical",
        mesh_grid_distance_ratio=2.0,
        level_refinement_factor=2,
        max_num_levels=3,
    )
    return datastore


def _build_model(model_cls, datastore, processor_layers):
    """Construct a hierarchical model with the given processor layer count."""
    return model_cls(
        datastore=datastore,
        graph_name="hierarchical",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=processor_layers,
    )


def _num_processor_steps(model):
    """Count the configured processor steps for either hierarchical model."""
    if isinstance(model, HiLAM):
        return len(model.mesh_down_gnns)
    processor = model.processor
    return len(processor) if hasattr(processor, "__len__") else 0


@pytest.mark.parametrize("model_cls", HIERARCHICAL_MODELS)
def test_negative_processor_layers_raises_value_error(
    model_cls, hierarchical_datastore
):
    """A negative layer count must fail with a descriptive ``ValueError``."""
    with pytest.raises(
        ValueError, match=r"processor_layers must be >= 0, got -1"
    ):
        _build_model(model_cls, hierarchical_datastore, -1)


@pytest.mark.parametrize("model_cls", HIERARCHICAL_MODELS)
def test_zero_processor_layers_is_supported(model_cls, hierarchical_datastore):
    """Zero layers must construct a model with no processor steps."""
    model = _build_model(model_cls, hierarchical_datastore, 0)

    assert _num_processor_steps(model) == 0


@pytest.mark.parametrize("model_cls", HIERARCHICAL_MODELS)
def test_positive_processor_layers_builds_requested_steps(
    model_cls, hierarchical_datastore
):
    """A positive layer count must build exactly that many processor steps."""
    model = _build_model(model_cls, hierarchical_datastore, 2)

    assert _num_processor_steps(model) == 2
