"""Convert between neural-lam's graph tensor-dict and ``pyg.HeteroData``.

The tensor-dict is the representation returned by
:func:`neural_lam.utils.load_graph` (a dictionary of edge-index and
node/edge-feature tensors, see that function's docstring for the keys).

The ``HeteroData`` representation follows the node/edge-type convention used
in the reference implementations linked from issue #385
(``leifdenby/weatherduck`` and ``matschreiner/equicast``): the grid nodes,
which carry the physical data, are the ``"data"`` node type and the mesh
nodes, the latent representation the model processes on, are the ``"hidden"``
node type. The three graph components map onto typed edges:

- ``g2m`` (grid-to-mesh)  -> ``("data", "to", "hidden")``
- ``m2m`` (mesh-to-mesh)  -> ``("hidden", "to", "hidden")``
- ``m2g`` (mesh-to-grid)  -> ``("hidden", "to", "data")``

Only flat (single mesh level) graphs are supported for now; hierarchical
graphs raise :class:`NotImplementedError` and are handled in a follow-up.
"""

from __future__ import annotations

# Standard library
from typing import Any, Dict

# Third-party
from torch_geometric.data import HeteroData

# Local
from .buffer_list import BufferList

# Node/edge-type names, kept in one place so the naming convention (see the
# issue #385 discussion) can be changed without touching call sites.
GRID_NODE_TYPE = "data"
MESH_NODE_TYPE = "hidden"
G2M_EDGE_TYPE = (GRID_NODE_TYPE, "to", MESH_NODE_TYPE)
M2M_EDGE_TYPE = (MESH_NODE_TYPE, "to", MESH_NODE_TYPE)
M2G_EDGE_TYPE = (MESH_NODE_TYPE, "to", GRID_NODE_TYPE)

# (edge_type, edge_index dict key, edge_feature dict key) for the three
# always-present components, iterated by both conversion directions.
_FLAT_EDGE_SPEC = (
    (G2M_EDGE_TYPE, "g2m_edge_index", "g2m_features"),
    (M2M_EDGE_TYPE, "m2m_edge_index", "m2m_features"),
    (M2G_EDGE_TYPE, "m2g_edge_index", "m2g_features"),
)


def graph_dict_to_heterodata(
    graph_dict: Dict[str, Any],
    num_grid_nodes: int,
    hierarchical: bool = False,
) -> HeteroData:
    """Build a ``pyg.HeteroData`` from a neural-lam graph tensor-dict.

    Parameters
    ----------
    graph_dict : dict
        Graph tensors as returned by :func:`neural_lam.utils.load_graph`. For
        a flat graph ``m2m_edge_index``, ``m2m_features`` and
        ``mesh_static_features`` are single tensors (not lists).
    num_grid_nodes : int
        Number of grid ("data") nodes. Taken from the datastore rather than
        inferred from the edge indices, because the graph spec allows grid
        nodes that no edge connects to.
    hierarchical : bool, default False
        Whether the graph is hierarchical. Only ``False`` is supported for
        now.

    Returns
    -------
    torch_geometric.data.HeteroData
        Typed graph with ``"data"`` and ``"hidden"`` node types and the
        ``g2m`` / ``m2m`` / ``m2g`` edge types. Mesh node features are stored
        on ``graph["hidden"].x`` and edge features on ``edge_attr``. Tensors
        are the same objects as in ``graph_dict`` (no copy).

    Raises
    ------
    NotImplementedError
        If ``hierarchical`` is ``True``.
    """
    if hierarchical:
        raise NotImplementedError(
            "graph_dict_to_heterodata currently supports only flat "
            "(single-level) graphs; hierarchical support is a follow-up."
        )

    graph = HeteroData()

    # Grid ("data") nodes have no stored static features here (grid features
    # are dynamic and passed at forward time), so only the count is set.
    graph[GRID_NODE_TYPE].num_nodes = int(num_grid_nodes)

    # Mesh ("hidden") nodes carry their static coordinate features.
    graph[MESH_NODE_TYPE].x = graph_dict["mesh_static_features"]

    for edge_type, edge_index_key, edge_feature_key in _FLAT_EDGE_SPEC:
        graph[edge_type].edge_index = graph_dict[edge_index_key]
        graph[edge_type].edge_attr = graph_dict[edge_feature_key]

    return graph


def heterodata_to_graph_dict(
    graph: HeteroData,
    hierarchical: bool = False,
) -> Dict[str, Any]:
    """Reconstruct a neural-lam graph tensor-dict from a ``HeteroData``.

    Inverse of :func:`graph_dict_to_heterodata`. The returned dictionary has
    the exact structure that :func:`neural_lam.utils.load_graph` produces for
    a flat graph, so it can be consumed by the model unpacking logic
    unchanged: single tensors for the mesh/edge entries and empty
    :class:`~neural_lam.utils.buffer_list.BufferList` objects for the
    (unused) hierarchical up/down entries.

    Parameters
    ----------
    graph : torch_geometric.data.HeteroData
        Graph as produced by :func:`graph_dict_to_heterodata`.
    hierarchical : bool, default False
        Whether the graph is hierarchical. Only ``False`` is supported for
        now.

    Returns
    -------
    dict
        Graph tensor-dict with the same 11 keys as
        :func:`neural_lam.utils.load_graph`.

    Raises
    ------
    NotImplementedError
        If ``hierarchical`` is ``True``.
    """
    if hierarchical:
        raise NotImplementedError(
            "heterodata_to_graph_dict currently supports only flat "
            "(single-level) graphs; hierarchical support is a follow-up."
        )

    graph_dict: Dict[str, Any] = {
        "mesh_static_features": graph[MESH_NODE_TYPE].x,
        # No inter-level edges for a flat graph.
        "mesh_up_edge_index": BufferList([], persistent=False),
        "mesh_down_edge_index": BufferList([], persistent=False),
        "mesh_up_features": BufferList([], persistent=False),
        "mesh_down_features": BufferList([], persistent=False),
    }
    for edge_type, edge_index_key, edge_feature_key in _FLAT_EDGE_SPEC:
        graph_dict[edge_index_key] = graph[edge_type].edge_index
        graph_dict[edge_feature_key] = graph[edge_type].edge_attr

    return graph_dict
