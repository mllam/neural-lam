"""Tests for static edge and mesh embedding caching in AR rollouts."""

# Standard library
import copy
from unittest.mock import patch

# Third-party
import pytest
import torch

# First-party
from neural_lam import api
from neural_lam import config as nlconfig
from neural_lam.models.forecasters.autoregressive import ARForecaster
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM
from neural_lam.models.step_predictors.graph.hi_lam import HiLAM
from tests.dummy_datastore import DummyDatastore


@pytest.fixture(scope="module")
def shared_env():
    """Module-scoped datastore and graphs fixture."""
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )
    api.create_graph(
        datastore=datastore,
        name="1level",
        archetype="keisler",
    )
    api.create_graph(
        datastore=datastore,
        name="hierarchical",
        archetype="hierarchical",
        level_refinement_factor=2,
        mesh_grid_distance_ratio=1.5,
    )
    return datastore, config


@pytest.mark.parametrize(
    ("model_class", "graph_name"),
    [
        (GraphLAM, "1level"),
        (HiLAM, "hierarchical"),
    ],
)
def test_static_edge_caching_numerical_parity(
    shared_env, model_class, graph_name
):
    """Verify rollout outputs and gradients match with and without caching."""
    torch.manual_seed(42)
    datastore, _ = shared_env

    predictor_cached = model_class(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=8,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=True,
    )
    predictor_uncached = copy.deepcopy(predictor_cached)

    forecaster_cached = ARForecaster(predictor_cached, datastore)

    b_size = 2
    pred_steps = 4
    num_grid = datastore.num_grid_points
    num_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    init_states = torch.randn(b_size, 2, num_grid, num_state)
    forcing = torch.randn(b_size, pred_steps, num_grid, d_forcing)
    boundary = torch.randn(b_size, pred_steps, num_grid, num_state)

    # 1. Unroll with forecaster (static embedding caching enabled)
    pred_cached, std_cached = forecaster_cached(init_states, forcing, boundary)
    loss_cached = pred_cached.sum() + std_cached.sum()
    loss_cached.backward()

    # 2. Manual unroll without static caching (static_embeddings=None)
    prev_prev_state = init_states[:, 0]
    prev_state = init_states[:, 1]
    pred_uncached_list = []
    std_uncached_list = []

    for i in range(pred_steps):
        step_forcing = forcing[:, i]
        step_boundary = boundary[:, i]
        step_pred, step_std = predictor_uncached(
            prev_state,
            prev_prev_state,
            step_forcing,
            static_embeddings=None,
        )
        new_state = (
            forecaster_cached.boundary_mask * step_boundary
            + forecaster_cached.interior_mask * step_pred
        )
        pred_uncached_list.append(new_state)
        std_uncached_list.append(step_std)
        prev_prev_state = prev_state
        prev_state = new_state

    pred_uncached = torch.stack(pred_uncached_list, dim=1)
    std_uncached = torch.stack(std_uncached_list, dim=1)
    loss_uncached = pred_uncached.sum() + std_uncached.sum()
    loss_uncached.backward()

    # Verify forward parity
    assert torch.allclose(pred_cached, pred_uncached, atol=1e-6)
    assert torch.allclose(std_cached, std_uncached, atol=1e-6)

    # Verify gradient parity across all trainable parameters
    for (name1, p1), (name2, p2) in zip(
        predictor_cached.named_parameters(),
        predictor_uncached.named_parameters(),
        strict=True,
    ):
        assert name1 == name2
        assert p1.grad is not None
        assert p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
            f"Gradient mismatch for parameter: {name1}"
        )


@pytest.mark.parametrize(
    ("model_class", "graph_name"),
    [
        (GraphLAM, "1level"),
        (HiLAM, "hierarchical"),
    ],
)
def test_every_embedder_called_only_once_per_rollout(
    shared_env, model_class, graph_name
):
    """Verify every static embedder executes exactly once per rollout."""
    datastore, _ = shared_env
    predictor = model_class(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=8,
        hidden_layers=1,
        processor_layers=1,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    forecaster = ARForecaster(predictor, datastore)

    b_size = 1
    pred_steps = 4
    num_grid = datastore.num_grid_points
    num_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    init_states = torch.randn(b_size, 2, num_grid, num_state)
    forcing = torch.randn(b_size, pred_steps, num_grid, d_forcing)
    boundary = torch.randn(b_size, pred_steps, num_grid, num_state)

    # Collect all static embedders for the model architecture
    embedders = [predictor.g2m_embedder, predictor.m2g_embedder]
    if isinstance(predictor, GraphLAM):
        embedders.extend([predictor.m2m_embedder, predictor.mesh_embedder])
    else:
        embedders.extend(predictor.mesh_embedders)
        embedders.extend(predictor.mesh_same_embedders)
        embedders.extend(predictor.mesh_up_embedders)
        embedders.extend(predictor.mesh_down_embedders)

    # Patch forward on every embedder
    patches = [
        patch.object(emb, "forward", wraps=emb.forward) for emb in embedders
    ]
    active_mocks = [p.start() for p in patches]
    try:
        forecaster(init_states, forcing, boundary)
        for m in active_mocks:
            assert m.call_count == 1
    finally:
        for p in patches:
            p.stop()


@pytest.mark.parametrize(
    ("model_class", "graph_name"),
    [
        (GraphLAM, "1level"),
        (HiLAM, "hierarchical"),
    ],
)
def test_standalone_fallback_behavior(shared_env, model_class, graph_name):
    """Verify predictor calls without static_embeddings compute on the fly."""
    datastore, _ = shared_env
    predictor = model_class(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=8,
        hidden_layers=1,
        processor_layers=1,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=True,
    )

    b_size = 1
    num_grid = datastore.num_grid_points
    num_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    prev_prev = torch.randn(b_size, num_grid, num_state)
    prev = torch.randn(b_size, num_grid, num_state)
    forcing = torch.randn(b_size, num_grid, d_forcing)

    # 1. Direct call without static_embeddings (defaults to None, on the fly)
    out_default, std_default = predictor(prev, prev_prev, forcing)

    # 2. Call with precomputed static embeddings
    static_emb = predictor.embed_static_features()
    out_precomputed, std_precomputed = predictor(
        prev, prev_prev, forcing, static_embeddings=static_emb
    )

    assert torch.allclose(out_default, out_precomputed, atol=1e-6)
    if std_default is not None:
        assert std_precomputed is not None
        assert torch.allclose(std_default, std_precomputed, atol=1e-6)
