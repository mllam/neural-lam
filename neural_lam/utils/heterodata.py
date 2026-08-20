"""Convert between neural-lam's graph tensor-dict and ``pyg.HeteroData``.

The tensor-dict is the representation returned by
:func:`neural_lam.utils.load_graph` (a dictionary of edge-index and
node/edge-feature tensors, see that function's docstring for the keys).

The ``HeteroData`` representation uses the ``"grid"`` and ``"mesh"`` node
types, matching the terminology used throughout the rest of ``neural-lam``
(``grid_static_features``, ``mesh_static_features``, ``g2m``/``m2m``/``m2g``).
For a flat (single mesh level) graph the three graph components map onto typed
edges:

- ``g2m`` (grid-to-mesh)  -> ``("grid", "to", "mesh")``
- ``m2m`` (mesh-to-mesh)  -> ``("mesh", "to", "mesh")``
- ``m2g`` (mesh-to-grid)  -> ``("mesh", "to", "grid")``

For a hierarchical graph (``L > 1`` mesh levels) each level is a distinct node
type ``"mesh_0"`` .. ``"mesh_{L-1}"`` (levels have different node counts and
their own embedders/GNNs in the model), and:

- ``g2m``/``m2g`` connect the grid to the bottom mesh level ``"mesh_0"``;
- intra-level edges are ``("mesh_i", "to", "mesh_i")``;
- inter-level edges are ``("mesh_i", "up", "mesh_{i+1}")`` and
  ``("mesh_{i+1}", "down", "mesh_i")`` for each of the ``L-1`` level pairs.

The node/edge-type names are defined as module-level constants / helpers so
the naming convention can be changed in one place (the reference
implementations in issue #385, ``leifdenby/weatherduck`` and
``matschreiner/equicast``, use ``"data"``/``"hidden"`` instead).
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


def mesh_level_node_type(level: int) -> str:
    """Return the node type for mesh ``level`` in a hierarchical graph."""
    return f"{MESH_NODE_TYPE}_{level}"


def _same_level_edge_type(level: int) -> tuple[str, str, str]:
    """Return the intra-level ``m2m`` edge type for mesh ``level``."""
    node_type = mesh_level_node_type(level)
    return (node_type, "to", node_type)


def _up_edge_type(level: int) -> tuple[str, str, str]:
    """Return the inter-level up edge type from ``level`` to ``level + 1``."""
    return (mesh_level_node_type(level), "up", mesh_level_node_type(level + 1))


def _down_edge_type(level: int) -> tuple[str, str, str]:
    """Return the inter-level down edge type from ``level + 1`` to ``level``."""
    return (
        mesh_level_node_type(level + 1),
        "down",
        mesh_level_node_type(level),
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
        ``mesh_static_features`` are single tensors; for a hierarchical graph
        they (and the ``mesh_up``/``mesh_down`` entries) are lists indexed by
        mesh level.
    num_grid_nodes : int
        Number of grid nodes. Taken from the datastore rather than inferred
        from the edge indices, because the graph spec allows grid nodes that
        no edge connects to.
    hierarchical : bool, default False
        Whether ``graph_dict`` describes a hierarchical (multi-level) graph.

    Returns
    -------
    torch_geometric.data.HeteroData
        Typed graph. Mesh node features are stored on the mesh node types'
        ``.x`` and edge features on ``edge_attr``. Tensors are the same
        objects as in ``graph_dict`` (no copy).
    """
    if hierarchical:
        return _hierarchical_graph_dict_to_heterodata(
            graph_dict, num_grid_nodes
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


def _hierarchical_graph_dict_to_heterodata(
    graph_dict: Dict[str, Any],
    num_grid_nodes: int,
) -> HeteroData:
    """Build a hierarchical ``HeteroData`` from a multi-level tensor-dict.

    Parameters
    ----------
    graph_dict : dict
        Hierarchical graph tensors from :func:`neural_lam.utils.load_graph`:
        ``mesh_static_features``, ``m2m_edge_index`` and ``m2m_features`` are
        length-``L`` lists and the ``mesh_up``/``mesh_down`` entries are
        length-``(L-1)`` lists.
    num_grid_nodes : int
        Number of grid nodes.

    Returns
    -------
    torch_geometric.data.HeteroData
        Graph with per-level mesh node types and typed intra-/inter-level
        edges.
    """
    graph = HeteroData()

    graph[GRID_NODE_TYPE].num_nodes = int(num_grid_nodes)

    mesh_static_features = graph_dict["mesh_static_features"]
    num_levels = len(mesh_static_features)
    for level in range(num_levels):
        graph[mesh_level_node_type(level)].x = mesh_static_features[level]

    # g2m/m2g connect the grid to the bottom mesh level only.
    bottom = mesh_level_node_type(0)
    graph[GRID_NODE_TYPE, "to", bottom].edge_index = graph_dict[
        "g2m_edge_index"
    ]
    graph[GRID_NODE_TYPE, "to", bottom].edge_attr = graph_dict["g2m_features"]
    graph[bottom, "to", GRID_NODE_TYPE].edge_index = graph_dict[
        "m2g_edge_index"
    ]
    graph[bottom, "to", GRID_NODE_TYPE].edge_attr = graph_dict["m2g_features"]

    # Intra-level (same-level) mesh edges, one entry per level.
    for level in range(num_levels):
        edge_type = _same_level_edge_type(level)
        graph[edge_type].edge_index = graph_dict["m2m_edge_index"][level]
        graph[edge_type].edge_attr = graph_dict["m2m_features"][level]

    # Inter-level up/down mesh edges, one entry per (level, level+1) pair.
    for level in range(num_levels - 1):
        up_type = _up_edge_type(level)
        graph[up_type].edge_index = graph_dict["mesh_up_edge_index"][level]
        graph[up_type].edge_attr = graph_dict["mesh_up_features"][level]

        down_type = _down_edge_type(level)
        graph[down_type].edge_index = graph_dict["mesh_down_edge_index"][level]
        graph[down_type].edge_attr = graph_dict["mesh_down_features"][level]

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
        Whether ``graph`` is a hierarchical (multi-level) graph.

    Returns
    -------
    dict
        The model's graph tensors, keyed by the names used in
        :func:`neural_lam.utils.load_graph`.
    """
    if hierarchical:
        return _hierarchical_graph_tensors_from_heterodata(graph)

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


def _hierarchical_graph_tensors_from_heterodata(
    graph: HeteroData,
) -> Dict[str, Any]:
    """Read the model's graph tensors out of a hierarchical ``HeteroData``.

    Parameters
    ----------
    graph : torch_geometric.data.HeteroData
        Graph as produced by :func:`_hierarchical_graph_dict_to_heterodata`.

    Returns
    -------
    dict
        The model's graph tensors, with the level-indexed entries collected
        per level into :class:`~neural_lam.utils.buffer_list.BufferList`
        objects, matching the hierarchical output of
        :func:`neural_lam.utils.load_graph`.
    """
    # Recover the number of mesh levels by counting the mesh node types.
    num_levels = 0
    while mesh_level_node_type(num_levels) in graph.node_types:
        num_levels += 1

    bottom = mesh_level_node_type(0)
    graph_dict: Dict[str, Any] = {
        "g2m_edge_index": graph[GRID_NODE_TYPE, "to", bottom].edge_index,
        "g2m_features": graph[GRID_NODE_TYPE, "to", bottom].edge_attr,
        "m2g_edge_index": graph[bottom, "to", GRID_NODE_TYPE].edge_index,
        "m2g_features": graph[bottom, "to", GRID_NODE_TYPE].edge_attr,
        "mesh_static_features": BufferList(
            [
                graph[mesh_level_node_type(level)].x
                for level in range(num_levels)
            ],
            persistent=False,
        ),
        "m2m_edge_index": BufferList(
            [
                graph[_same_level_edge_type(level)].edge_index
                for level in range(num_levels)
            ],
            persistent=False,
        ),
        "m2m_features": BufferList(
            [
                graph[_same_level_edge_type(level)].edge_attr
                for level in range(num_levels)
            ],
            persistent=False,
        ),
        "mesh_up_edge_index": BufferList(
            [
                graph[_up_edge_type(level)].edge_index
                for level in range(num_levels - 1)
            ],
            persistent=False,
        ),
        "mesh_up_features": BufferList(
            [
                graph[_up_edge_type(level)].edge_attr
                for level in range(num_levels - 1)
            ],
            persistent=False,
        ),
        "mesh_down_edge_index": BufferList(
            [
                graph[_down_edge_type(level)].edge_index
                for level in range(num_levels - 1)
            ],
            persistent=False,
        ),
        "mesh_down_features": BufferList(
            [
                graph[_down_edge_type(level)].edge_attr
                for level in range(num_levels - 1)
            ],
            persistent=False,
        ),
    }

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
