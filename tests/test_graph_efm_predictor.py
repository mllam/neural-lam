"""Unit tests for the Graph-EFM single-step probabilistic predictors.

These mirror the smoke-test pattern used for the deterministic predictors
(see ``tests/test_gnn_layers.py``): build the flat (GraphEFMMultiScale) and
hierarchical (GraphEFM) variants on the real example datastore with a freshly
created graph, then exercise ``forward`` on synthetic tensors.
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.step_predictors.graph.graph_efm import (
    GraphEFM,
    GraphEFMMultiScale,
)
from tests.conftest import init_datastore_example

NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def _datastore_with_graph(graph_name):
    """Create the example datastore and ensure ``graph_name`` exists."""
    datastore = init_datastore_example("mdp")

    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    else:
        hierarchical = False
        n_max_levels = 1

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )
    return datastore


def _build_predictor(graph_name, output_std=False):
    datastore = _datastore_with_graph(graph_name)
    if graph_name == "hierarchical":
        predictor_class = GraphEFM
        layer_kwargs = {
            "prior_intra_level_layers": 1,
            "encoder_intra_level_layers": 1,
            "decoder_intra_level_layers": 1,
        }
    else:
        predictor_class = GraphEFMMultiScale
        layer_kwargs = {
            "prior_m2m_layers": 1,
            "encoder_m2m_layers": 1,
            "decoder_m2m_layers": 1,
        }
    predictor = predictor_class(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        latent_dim=4,
        learn_prior=True,
        prior_dist="isotropic",
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=output_std,
        **layer_kwargs,
    )
    return predictor, datastore


def _make_inputs(predictor, datastore, batch_size=2):
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    prev_prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    forcing = torch.randn(batch_size, num_grid_nodes, d_forcing)
    return prev_state, prev_prev_state, forcing, d_state


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_shapes_and_no_std(graph_name):
    """forward returns a (B, num_grid_nodes, d_state) state and None std when
    output_std is False, for both flat and hierarchical graphs."""
    predictor, datastore = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )

    new_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert new_state.shape == (2, predictor.num_grid_nodes, d_state)
    assert pred_std is None


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_output_std_returns_std(graph_name):
    """With output_std=True the decoder produces a positive std of the same
    shape as the state."""
    predictor, datastore = _build_predictor(graph_name, output_std=True)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )

    new_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    expected = (2, predictor.num_grid_nodes, d_state)
    assert new_state.shape == expected
    assert pred_std is not None
    assert pred_std.shape == expected
    assert (pred_std > 0).all()


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_member_stochasticity(graph_name):
    """Two forward calls with identical inputs differ, because the latent is
    resampled from the prior each call (catches an unused-latent regression)."""
    predictor, datastore = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, _ = _make_inputs(predictor, datastore)

    out_a, _ = predictor(prev_state, prev_prev_state, forcing)
    out_b, _ = predictor(prev_state, prev_prev_state, forcing)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize(
    "predictor_class, graph_name",
    [(GraphEFM, "1level"), (GraphEFMMultiScale, "hierarchical")],
)
def test_graph_type_mismatch_raises(predictor_class, graph_name):
    """GraphEFM requires a hierarchical graph and GraphEFMMultiScale a flat one;
    constructing with the wrong graph type raises ValueError."""
    datastore = _datastore_with_graph(graph_name)
    with pytest.raises(ValueError, match="mesh graph"):
        predictor_class(
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=4,
            hidden_layers=1,
        )
