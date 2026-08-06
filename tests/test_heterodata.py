"""Tests for the graph tensor-dict <-> pyg.HeteroData conversion (issue #385).

Two layers of testing:

- unit tests on ``graph_dict_to_heterodata`` / ``graph_tensors_from_heterodata``
  with small hand-built tensor-dicts (no datastore/model needed), for both
  flat and hierarchical graphs;
- model-level equivalence tests showing that ``GraphLAM`` (flat) and
  ``HiLAM`` (hierarchical) built with ``use_heterodata=True`` are identical to
  ones built the existing way, in their parameters/buffers, forward output
  and training.
"""

# Standard library
from pathlib import Path

# Third-party
import torch
from torch_geometric.data import HeteroData

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import GraphLAM, HiLAM
from neural_lam.utils import (
    graph_dict_to_heterodata,
    graph_tensors_from_heterodata,
    load_and_register_graph,
)
from neural_lam.utils.buffer_list import BufferList
from neural_lam.utils.heterodata import (
    G2M_EDGE_TYPE,
    GRID_NODE_TYPE,
    M2G_EDGE_TYPE,
    M2M_EDGE_TYPE,
    MESH_NODE_TYPE,
    mesh_level_node_type,
)

# Keys that ``load_graph`` returns; used to assert the round-trip is exact.
_GRAPH_DICT_KEYS = {
    "g2m_edge_index",
    "m2g_edge_index",
    "m2m_edge_index",
    "mesh_up_edge_index",
    "mesh_down_edge_index",
    "g2m_features",
    "m2g_features",
    "m2m_features",
    "mesh_up_features",
    "mesh_down_features",
    "mesh_static_features",
}


def _flat_graph_dict(edge_feature_dim: int = 3, num_mesh: int = 4):
    """Build a minimal flat graph tensor-dict for the unit tests.

    Parameters
    ----------
    edge_feature_dim : int, default 3
        Number of edge feature columns (3 for 2D graphs, 4 for 3D).
    num_mesh : int, default 4
        Number of mesh nodes.

    Returns
    -------
    dict
        A tensor-dict with the same keys and structure as the flat output of
        :func:`neural_lam.utils.load_graph`.
    """

    def edges(n):
        return torch.tensor([[0, 1, 2], [1, 2, 0]][:2], dtype=torch.int64)[
            :, :n
        ]

    def feats(n):
        return torch.arange(n * edge_feature_dim, dtype=torch.float32).reshape(
            n, edge_feature_dim
        )

    return {
        "g2m_edge_index": torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.int64
        ),
        "m2g_edge_index": torch.tensor(
            [[0, 1, 2], [3, 4, 5]], dtype=torch.int64
        ),
        "m2m_edge_index": edges(3),
        "mesh_up_edge_index": BufferList([], persistent=False),
        "mesh_down_edge_index": BufferList([], persistent=False),
        "g2m_features": feats(3),
        "m2g_features": feats(3),
        "m2m_features": feats(3),
        "mesh_up_features": BufferList([], persistent=False),
        "mesh_down_features": BufferList([], persistent=False),
        "mesh_static_features": torch.arange(
            num_mesh * 2, dtype=torch.float32
        ).reshape(num_mesh, 2),
    }


def test_builder_flat_structure():
    """The builder produces the expected node/edge types and tensors."""
    graph_dict = _flat_graph_dict()
    num_grid = 6
    graph = graph_dict_to_heterodata(graph_dict, num_grid_nodes=num_grid)

    assert isinstance(graph, HeteroData)
    assert set(graph.node_types) == {GRID_NODE_TYPE, MESH_NODE_TYPE}
    assert set(graph.edge_types) == {
        G2M_EDGE_TYPE,
        M2M_EDGE_TYPE,
        M2G_EDGE_TYPE,
    }

    # Grid node count comes from the datastore, not from edge indices.
    assert graph[GRID_NODE_TYPE].num_nodes == num_grid
    # Mesh node features are stored on .x, unchanged.
    assert torch.equal(
        graph[MESH_NODE_TYPE].x, graph_dict["mesh_static_features"]
    )

    # Each component maps to the right typed edge, with identical tensors.
    for edge_type, ei_key, ef_key in (
        (G2M_EDGE_TYPE, "g2m_edge_index", "g2m_features"),
        (M2M_EDGE_TYPE, "m2m_edge_index", "m2m_features"),
        (M2G_EDGE_TYPE, "m2g_edge_index", "m2g_features"),
    ):
        assert torch.equal(graph[edge_type].edge_index, graph_dict[ei_key])
        assert torch.equal(graph[edge_type].edge_attr, graph_dict[ef_key])


def test_builder_roundtrip_is_exact():
    """dict -> HeteroData -> dict reproduces the original tensor-dict."""
    graph_dict = _flat_graph_dict()
    graph = graph_dict_to_heterodata(graph_dict, num_grid_nodes=6)
    restored = graph_tensors_from_heterodata(graph)

    assert set(restored.keys()) == _GRAPH_DICT_KEYS
    for key in _GRAPH_DICT_KEYS:
        original = graph_dict[key]
        if isinstance(original, torch.Tensor):
            assert torch.equal(restored[key], original), key
        else:
            # The unused hierarchical entries stay empty BufferLists.
            assert isinstance(restored[key], BufferList), key
            assert len(restored[key]) == 0, key


def test_builder_variable_edge_feature_dim():
    """Edge feature width is not hardcoded (e.g. 4 cols for 3D graphs)."""
    graph_dict = _flat_graph_dict(edge_feature_dim=4)
    graph = graph_dict_to_heterodata(graph_dict, num_grid_nodes=6)
    assert graph[G2M_EDGE_TYPE].edge_attr.shape[1] == 4
    assert graph[M2M_EDGE_TYPE].edge_attr.shape[1] == 4


def test_builder_grid_count_not_inferred_from_edges():
    """num_grid_nodes is taken from the argument, not from edge maxima."""
    graph_dict = _flat_graph_dict()
    # 100 is far larger than any index present in the edge tensors.
    graph = graph_dict_to_heterodata(graph_dict, num_grid_nodes=100)
    assert graph[GRID_NODE_TYPE].num_nodes == 100


def _hierarchical_graph_dict(num_levels: int = 3, edge_feature_dim: int = 3):
    """Build a minimal hierarchical graph tensor-dict for the unit tests.

    Level-indexed entries are :class:`BufferList` objects (as ``load_graph``
    returns for hierarchical graphs), and each level's tensors carry a
    level-dependent offset so the tests can detect any mislabelling of levels.

    Parameters
    ----------
    num_levels : int, default 3
        Number of mesh levels ``L``.
    edge_feature_dim : int, default 3
        Number of edge feature columns.

    Returns
    -------
    dict
        A tensor-dict with the same keys/structure as the hierarchical output
        of :func:`neural_lam.utils.load_graph`.
    """
    mesh_counts = [4, 3, 2, 2][:num_levels]

    def ei(n_edges):
        cols = [[i % 2, (i + 1) % 2] for i in range(n_edges)]
        return torch.tensor(cols, dtype=torch.int64).t().contiguous()

    def ef(n_edges, tag):
        base = torch.arange(n_edges * edge_feature_dim, dtype=torch.float32)
        return (base + 100 * tag).reshape(n_edges, edge_feature_dim)

    def mesh_x(n_nodes, tag):
        base = torch.arange(n_nodes * 2, dtype=torch.float32)
        return (base + 1000 * tag).reshape(n_nodes, 2)

    def bl(tensors):
        return BufferList(tensors, persistent=False)

    return {
        "g2m_edge_index": torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.int64
        ),
        "m2g_edge_index": torch.tensor(
            [[0, 1, 2], [3, 4, 5]], dtype=torch.int64
        ),
        "g2m_features": ef(3, tag=1),
        "m2g_features": ef(3, tag=2),
        "mesh_static_features": bl(
            [mesh_x(mesh_counts[lvl], tag=lvl) for lvl in range(num_levels)]
        ),
        "m2m_edge_index": bl([ei(3) for _ in range(num_levels)]),
        "m2m_features": bl([ef(3, tag=10 + lvl) for lvl in range(num_levels)]),
        "mesh_up_edge_index": bl([ei(2) for _ in range(num_levels - 1)]),
        "mesh_up_features": bl(
            [ef(2, tag=20 + lvl) for lvl in range(num_levels - 1)]
        ),
        "mesh_down_edge_index": bl([ei(2) for _ in range(num_levels - 1)]),
        "mesh_down_features": bl(
            [ef(2, tag=30 + lvl) for lvl in range(num_levels - 1)]
        ),
    }


def test_builder_hierarchical_structure():
    """The hierarchical builder produces per-level node/edge types."""
    num_levels = 3
    graph_dict = _hierarchical_graph_dict(num_levels=num_levels)
    graph = graph_dict_to_heterodata(
        graph_dict, num_grid_nodes=6, hierarchical=True
    )

    # One grid node type + one node type per mesh level.
    expected_nodes = {GRID_NODE_TYPE} | {
        mesh_level_node_type(lvl) for lvl in range(num_levels)
    }
    assert set(graph.node_types) == expected_nodes

    # g2m/m2g connect the grid to the bottom mesh level only.
    bottom = mesh_level_node_type(0)
    assert (GRID_NODE_TYPE, "to", bottom) in graph.edge_types
    assert (bottom, "to", GRID_NODE_TYPE) in graph.edge_types

    # Per-level mesh features land on the right level, in order.
    for lvl in range(num_levels):
        assert torch.equal(
            graph[mesh_level_node_type(lvl)].x,
            graph_dict["mesh_static_features"][lvl],
        )
        same = (mesh_level_node_type(lvl), "to", mesh_level_node_type(lvl))
        assert torch.equal(
            graph[same].edge_attr, graph_dict["m2m_features"][lvl]
        )

    # Inter-level up/down edges for each of the L-1 pairs.
    for lvl in range(num_levels - 1):
        up = (mesh_level_node_type(lvl), "up", mesh_level_node_type(lvl + 1))
        down = (
            mesh_level_node_type(lvl + 1),
            "down",
            mesh_level_node_type(lvl),
        )
        assert torch.equal(
            graph[up].edge_attr, graph_dict["mesh_up_features"][lvl]
        )
        assert torch.equal(
            graph[down].edge_attr, graph_dict["mesh_down_features"][lvl]
        )


def test_builder_hierarchical_roundtrip_is_exact():
    """dict -> HeteroData -> dict reproduces the hierarchical tensor-dict."""
    graph_dict = _hierarchical_graph_dict(num_levels=3)
    graph = graph_dict_to_heterodata(
        graph_dict, num_grid_nodes=6, hierarchical=True
    )
    restored = graph_tensors_from_heterodata(graph, hierarchical=True)

    assert set(restored.keys()) == _GRAPH_DICT_KEYS
    for key in _GRAPH_DICT_KEYS:
        original = graph_dict[key]
        if isinstance(original, torch.Tensor):
            assert torch.equal(restored[key], original), key
        else:
            # Level-indexed entries stay lists of the same length, per level.
            assert len(restored[key]) == len(original), key
            for lvl, (a, b) in enumerate(zip(restored[key], original)):
                assert torch.equal(a, b), f"{key}[{lvl}]"


def test_load_and_register_graph_builds_hierarchical_heterodata():
    """Loading a hierarchical graph gives a per-level ``HeteroData`` object.

    ``load_and_register_graph`` is where the graph is loaded and the
    ``HeteroData`` object is built, and the only place that knows whether the
    graph is hierarchical, so check the hierarchical case directly on it.
    """
    # First-party
    from tests.dummy_datastore import DummyDatastore

    datastore = DummyDatastore()
    graph_name = "hierarchical3"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=3,
            hierarchical=True,
        )

    module = torch.nn.Module()
    hierarchical = load_and_register_graph(
        module,
        datastore,
        graph_name,
        mesh_node_features_scaling=1.0,
        use_heterodata=True,
    )

    assert hierarchical
    assert isinstance(module.graph, HeteroData)
    # One node type per mesh level, and the registered per-level tensors
    # match the levels held on the HeteroData object.
    num_levels = len(module.mesh_static_features)
    assert num_levels > 1
    for level in range(num_levels):
        assert torch.equal(
            module.graph[mesh_level_node_type(level)].x,
            module.mesh_static_features[level],
        )


def _build_graphlam(datastore, graph_name, use_heterodata):
    """Construct a small GraphLAM with a fixed seed for equivalence tests."""
    torch.manual_seed(0)
    return GraphLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=False,
        use_heterodata=use_heterodata,
    )


def test_graphlam_heterodata_equivalence(tmp_path):
    """GraphLAM(use_heterodata=True) is identical to the dict-based model.

    Directly addresses the issue #385 requirement that training proceed
    identically with the existing and the new HeteroData datastructure:
    identical parameters, identical graph buffers, identical forward output
    for the same inputs, and, when trained with the same optimizer steps,
    identical per-step losses and identical weights afterwards.
    """
    # First-party
    from tests.dummy_datastore import DummyDatastore

    datastore = DummyDatastore()
    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    model_dict = _build_graphlam(datastore, graph_name, use_heterodata=False)
    model_hd = _build_graphlam(datastore, graph_name, use_heterodata=True)

    # The HeteroData model exposes the graph as a HeteroData object.
    assert isinstance(model_hd.graph, HeteroData)
    assert not hasattr(model_dict, "graph")

    # Identical parameters.
    params_dict = dict(model_dict.named_parameters())
    params_hd = dict(model_hd.named_parameters())
    assert params_dict.keys() == params_hd.keys()
    for name in params_dict:
        assert torch.equal(params_dict[name], params_hd[name]), name

    # Identical graph buffers (edge indices + features + mesh features).
    for name in (
        "g2m_edge_index",
        "m2g_edge_index",
        "m2m_edge_index",
        "g2m_features",
        "m2g_features",
        "m2m_features",
        "mesh_static_features",
    ):
        assert torch.equal(
            getattr(model_dict, name), getattr(model_hd, name)
        ), name

    # Identical forward output and identical training.
    _assert_forward_and_training_equivalent(model_dict, model_hd)


def _assert_forward_and_training_equivalent(model_dict, model_hd):
    """Assert two step-predictor models forward and train identically.

    Runs an identical forward pass (identical output) and then a few identical
    optimizer steps (identical per-step losses and identical weights
    afterwards) on both models.

    Parameters
    ----------
    model_dict : neural_lam.models.step_predictors.base.BaseStepPredictor
        The reference model (dict-based graph).
    model_hd : neural_lam.models.step_predictors.base.BaseStepPredictor
        The HeteroData-based model to compare against.
    """
    num_grid = model_dict.num_grid_nodes
    d_state = model_dict.grid_output_dim
    forcing_dim = (
        model_dict.grid_input_dim
        - 2 * d_state
        - (model_dict.grid_static_features.shape[1])
    )
    torch.manual_seed(1)
    prev_state = torch.randn(1, num_grid, d_state)
    prev_prev_state = torch.randn(1, num_grid, d_state)
    forcing = torch.randn(1, num_grid, max(forcing_dim, 0))

    model_dict.eval()
    model_hd.eval()
    with torch.no_grad():
        out_dict, _ = model_dict(prev_state, prev_prev_state, forcing)
        out_hd, _ = model_hd(prev_state, prev_prev_state, forcing)

    assert torch.equal(out_dict, out_hd)

    # Training proceeds identically: running the same optimizer steps on both
    # models yields identical losses at every step and identical weights
    # afterwards.
    model_dict.train()
    model_hd.train()
    target = torch.randn(1, num_grid, d_state)
    opt_dict = torch.optim.SGD(model_dict.parameters(), lr=0.1)
    opt_hd = torch.optim.SGD(model_hd.parameters(), lr=0.1)
    for _ in range(3):
        step_losses = []
        for model, optimizer in ((model_dict, opt_dict), (model_hd, opt_hd)):
            optimizer.zero_grad()
            pred, _ = model(prev_state, prev_prev_state, forcing)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            step_losses.append(loss)
        assert torch.equal(step_losses[0], step_losses[1])

    trained_dict = dict(model_dict.named_parameters())
    trained_hd = dict(model_hd.named_parameters())
    for name in trained_dict:
        assert torch.equal(trained_dict[name], trained_hd[name]), name


def _flatten_graph_attr(value):
    """Return graph buffer(s) as a list of tensors (tensor or BufferList)."""
    if isinstance(value, torch.Tensor):
        return [value]
    return list(value)


def _build_hilam(datastore, graph_name, use_heterodata):
    """Construct a small HiLAM with a fixed seed for equivalence tests."""
    torch.manual_seed(0)
    return HiLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=False,
        use_heterodata=use_heterodata,
    )


def test_hilam_heterodata_equivalence(tmp_path):
    """HiLAM(use_heterodata=True) is identical to the dict-based model.

    The hierarchical counterpart of the GraphLAM test: builds a multi-level
    graph and asserts identical parameters, identical per-level graph buffers,
    identical forward output and identical training (per-step losses and
    weights) with and without the HeteroData datastructure.
    """
    # First-party
    from tests.dummy_datastore import DummyDatastore

    datastore = DummyDatastore()
    graph_name = "hierarchical3"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=3,
            hierarchical=True,
        )

    model_dict = _build_hilam(datastore, graph_name, use_heterodata=False)
    model_hd = _build_hilam(datastore, graph_name, use_heterodata=True)

    # The HeteroData model is hierarchical and exposes per-level mesh types.
    assert model_dict.hierarchical and model_hd.hierarchical
    assert isinstance(model_hd.graph, HeteroData)
    for lvl in range(model_hd.num_levels):
        assert mesh_level_node_type(lvl) in model_hd.graph.node_types

    # Identical parameters.
    params_dict = dict(model_dict.named_parameters())
    params_hd = dict(model_hd.named_parameters())
    assert params_dict.keys() == params_hd.keys()
    for name in params_dict:
        assert torch.equal(params_dict[name], params_hd[name]), name

    # Identical graph buffers, including the per-level BufferList entries.
    for name in (
        "g2m_edge_index",
        "m2g_edge_index",
        "m2m_edge_index",
        "g2m_features",
        "m2g_features",
        "m2m_features",
        "mesh_static_features",
        "mesh_up_edge_index",
        "mesh_down_edge_index",
        "mesh_up_features",
        "mesh_down_features",
    ):
        tensors_dict = _flatten_graph_attr(getattr(model_dict, name))
        tensors_hd = _flatten_graph_attr(getattr(model_hd, name))
        assert len(tensors_dict) == len(tensors_hd), name
        for lvl, (a, b) in enumerate(zip(tensors_dict, tensors_hd)):
            assert torch.equal(a, b), f"{name}[{lvl}]"

    # Identical forward output and identical training.
    _assert_forward_and_training_equivalent(model_dict, model_hd)
