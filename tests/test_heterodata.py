"""Tests for the graph tensor-dict <-> pyg.HeteroData conversion (issue #385).

Two layers of testing:

- unit tests on ``graph_dict_to_heterodata`` / ``heterodata_to_graph_dict``
  with small hand-built tensor-dicts (no datastore/model needed);
- a model-level equivalence test showing that a ``GraphLAM`` built with
  ``use_heterodata=True`` is identical to one built the existing way, both in
  its parameters/buffers and in its forward output.
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import torch
from torch_geometric.data import HeteroData

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import GraphLAM
from neural_lam.utils import (
    graph_dict_to_heterodata,
    heterodata_to_graph_dict,
)
from neural_lam.utils.buffer_list import BufferList
from neural_lam.utils.heterodata import (
    G2M_EDGE_TYPE,
    GRID_NODE_TYPE,
    M2G_EDGE_TYPE,
    M2M_EDGE_TYPE,
    MESH_NODE_TYPE,
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
    restored = heterodata_to_graph_dict(graph)

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


def test_builder_rejects_hierarchical():
    """Hierarchical graphs are explicitly not supported yet."""
    graph_dict = _flat_graph_dict()
    with pytest.raises(NotImplementedError):
        graph_dict_to_heterodata(
            graph_dict, num_grid_nodes=6, hierarchical=True
        )
    graph = graph_dict_to_heterodata(graph_dict, num_grid_nodes=6)
    with pytest.raises(NotImplementedError):
        heterodata_to_graph_dict(graph, hierarchical=True)


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
    identical parameters, identical graph buffers, and identical forward
    output for the same inputs.
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

    # Identical forward output for the same inputs.
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
