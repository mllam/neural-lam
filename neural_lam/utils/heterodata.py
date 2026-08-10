"""Convert between neural-lam's graph tensor-dict and ``pyg.HeteroData``.

The tensor-dict is the representation returned by
:func:`neural_lam.utils.load_graph` (a dictionary of edge-index and
node/edge-feature tensors, see that function's docstring for the keys).

The ``HeteroData`` representation uses the ``"grid"`` and ``"mesh"`` node
types, matching the terminology used throughout the rest of ``neural-lam``
(``grid_static_features``, ``mesh_static_features``, ``g2m``/``m2m``/``m2g``).
The three graph components map onto typed edges:

- ``g2m`` (grid-to-mesh)  -> ``("grid", "to", "mesh")``
- ``m2m`` (mesh-to-mesh)  -> ``("mesh", "to", "mesh")``
- ``m2g`` (mesh-to-grid)  -> ``("mesh", "to", "grid")``

The node/edge-type names are defined as module-level constants so the naming
convention can be changed in one place (the reference implementations in
issue #385, ``leifdenby/weatherduck`` and ``matschreiner/equicast``, use
``"data"``/``"hidden"`` instead).

Only flat (single mesh level) graphs are supported for now; hierarchical
graphs raise :class:`NotImplementedError` and are handled in a follow-up.
"""

from __future__ import annotations

# Standard library
from typing import Any, Dict

# Third-party
import torch
from torch_geometric.data import HeteroData

# Local
from .buffer_list import BufferList

# Node/edge-type names, kept in one place so the naming convention (see the
# issue #385 discussion) can be changed without touching call sites.
GRID_NODE_TYPE = "grid"
MESH_NODE_TYPE = "mesh"
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

# Names the graph tensors are registered under on a module, i.e. the keys of
# the tensor-dict returned by :func:`neural_lam.utils.load_graph`.
GRAPH_TENSOR_NAMES = (
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
        Number of grid nodes. Taken from the datastore rather than inferred
        from the edge indices, because the graph spec allows grid nodes that
        no edge connects to.
    hierarchical : bool, default False
        Whether the graph is hierarchical. Only ``False`` is supported for
        now.

    Returns
    -------
    torch_geometric.data.HeteroData
        Typed graph with ``"grid"`` and ``"mesh"`` node types and the
        ``g2m`` / ``m2m`` / ``m2g`` edge types. Mesh node features are stored
        on ``graph["mesh"].x`` and edge features on ``edge_attr``. Tensors
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

    # Grid nodes have no stored static features here (grid features are
    # dynamic and passed at forward time), so only the count is set.
    graph[GRID_NODE_TYPE].num_nodes = int(num_grid_nodes)

    # Mesh nodes carry their static coordinate features.
    graph[MESH_NODE_TYPE].x = graph_dict["mesh_static_features"]

    for edge_type, edge_index_key, edge_feature_key in _FLAT_EDGE_SPEC:
        graph[edge_type].edge_index = graph_dict[edge_index_key]
        graph[edge_type].edge_attr = graph_dict[edge_feature_key]

    return graph


def graph_tensors_from_heterodata(
    graph: HeteroData,
    hierarchical: bool = False,
) -> Dict[str, Any]:
    """Read the model's graph tensors out of a ``HeteroData`` object.

    This is how the model obtains its graph tensors when it represents the
    graph as a ``HeteroData``: every tensor is looked up on the typed
    node/edge stores of ``graph``. They are returned under the names the
    model refers to them by (the same names
    :func:`neural_lam.utils.load_graph` uses), so only the *source* of the
    tensors changes, not the rest of the model's setup. It is also the exact
    inverse of :func:`graph_dict_to_heterodata`.

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
        The model's graph tensors, keyed by the names used in
        :func:`neural_lam.utils.load_graph`.

    Raises
    ------
    NotImplementedError
        If ``hierarchical`` is ``True``.
    """
    if hierarchical:
        raise NotImplementedError(
            "graph_tensors_from_heterodata currently supports only flat "
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


def heterodata_from_module(
    module: torch.nn.Module,
    num_grid_nodes: int,
    hierarchical: bool = False,
) -> HeteroData:
    """Build a ``HeteroData`` view of the graph tensors held by ``module``.

    The object is built on demand from the graph tensors currently registered
    on ``module``, rather than being stored on it. A stored ``HeteroData``
    would not be moved by :meth:`torch.nn.Module._apply`, since it is neither
    a tensor, parameter nor module, so it would keep referring to the
    pre-move tensors after e.g. ``.to(device)`` and silently disagree with the
    module's buffers. Building it on access keeps it consistent with whatever
    device and dtype the module currently holds, and adds no copies: the
    returned object references the module's tensors.

    Parameters
    ----------
    module : torch.nn.Module
        Module the graph tensors were registered on, e.g. by
        :func:`neural_lam.utils.load_and_register_graph`.
    num_grid_nodes : int
        Number of grid nodes.
    hierarchical : bool, default False
        Whether the graph is hierarchical.

    Returns
    -------
    torch_geometric.data.HeteroData
        Typed graph referencing ``module``'s current graph tensors.
    """
    graph_dict = {name: getattr(module, name) for name in GRAPH_TENSOR_NAMES}
    return graph_dict_to_heterodata(
        graph_dict,
        num_grid_nodes=num_grid_nodes,
        hierarchical=hierarchical,
    )
