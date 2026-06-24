"""Loading and zero-indexing of mesh/grid graph tensors."""

# Standard library
import os
from pathlib import Path
from typing import Any, Union

# Third-party
import torch

# Local
from .buffer_list import BufferList


def zero_index_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Make both sender and receiver indices of edge_index start at 0.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor of shape (2, num_edges).

    Returns
    -------
    torch.Tensor
        Edge index tensor with indices starting at 0.
    """
    return edge_index - edge_index.min(dim=1, keepdim=True)[0]


def zero_index_m2g(
    m2g_edge_index: torch.Tensor,
    mesh_static_features: list[torch.Tensor],
    mesh_first: bool,
    restore: bool = False,
) -> torch.Tensor:
    """
    Zero-index the m2g (mesh-to-grid) edge index, or undo this operation.

    Special handling is needed since not all mesh nodes may be present.

    Parameters
    ----------
    m2g_edge_index : torch.Tensor
        Edge index tensor of shape (2, num_edges).
    mesh_static_features : list of torch.Tensor
        Mesh node feature tensors.
    mesh_first : bool
        If True, mesh nodes are indexed before grid nodes.
    restore : bool
        If True, undo zero-indexing (restore original indices).

    Returns
    -------
    torch.Tensor
        Edge index tensor with zero-based or restored indices.
    """

    sign = 1 if restore else -1

    if mesh_first:
        # Mesh has the first indices, adjust grid indices (row 1).
        # Use the total number of mesh nodes across all levels because
        # create_graph offsets grid nodes by the full mesh node count.
        num_mesh_nodes = sum(sf.shape[0] for sf in mesh_static_features)
        return torch.stack(
            (
                m2g_edge_index[0],
                m2g_edge_index[1] + sign * num_mesh_nodes,
            ),
            dim=0,
        )
    else:
        # Grid (interior) has the first indices, adjust mesh indices (row 0)
        num_interior_nodes = m2g_edge_index[1].max() + 1
        return torch.stack(
            (
                m2g_edge_index[0] + sign * num_interior_nodes,
                m2g_edge_index[1],
            ),
            dim=0,
        )


def zero_index_g2m(
    g2m_edge_index: torch.Tensor,
    mesh_static_features: list[torch.Tensor],
    mesh_first: bool,
    restore: bool = False,
) -> torch.Tensor:
    """
    Zero-index the g2m (grid-to-mesh) edge index, or undo this operation.

    Special handling is needed since not all mesh nodes may be present.

    Parameters
    ----------
    g2m_edge_index : torch.Tensor
        Edge index tensor of shape (2, num_edges).
    mesh_static_features : list of torch.Tensor
        Mesh node feature tensors.
    mesh_first : bool
        If True, mesh nodes are indexed before grid nodes.
    restore : bool
        If True, undo zero-indexing (restore original indices).

    Returns
    -------
    torch.Tensor
        Edge index tensor with zero-based or restored indices.
    """

    sign = 1 if restore else -1

    if mesh_first:
        # Mesh has the first indices, adjust grid indices (row 0).
        # Use the total number of mesh nodes across all levels because
        # create_graph offsets grid nodes by the full mesh node count.
        num_mesh_nodes = sum(sf.shape[0] for sf in mesh_static_features)
        return torch.stack(
            (
                g2m_edge_index[0] + sign * num_mesh_nodes,
                g2m_edge_index[1],
            ),
            dim=0,
        )
    else:
        # Grid has the first indices, adjust mesh indices (row 1)
        num_grid_nodes = g2m_edge_index[0].max() + 1
        return torch.stack(
            (
                g2m_edge_index[0],
                g2m_edge_index[1] + sign * num_grid_nodes,
            ),
            dim=0,
        )


def load_graph(
    graph_dir_path: Union[str, Path], device: str = "cpu"
) -> tuple[bool, dict[str, Any]]:
    """Load all tensors representing the graph from `graph_dir_path`.

    Needs the following files for all graphs:
    - m2m_edge_index.pt
    - g2m_edge_index.pt
    - m2g_edge_index.pt
    - m2m_features.pt
    - g2m_features.pt
    - m2g_features.pt
    - mesh_features.pt

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
    - mesh_down_edge_index.pt
    - mesh_up_features.pt
    - mesh_down_features.pt

    Parameters
    ----------
    graph_dir_path : str
        Path to directory containing the graph files.
    device : str
        Device to load tensors to.

    Returns
    -------
    hierarchical : bool
        Whether the graph is hierarchical.
    graph : dict
        Dictionary containing the graph tensors, with keys as follows:
        - g2m_edge_index
        - m2g_edge_index
        - m2m_edge_index
        - mesh_up_edge_index
        - mesh_down_edge_index
        - g2m_features
        - m2g_features
        - m2m_features
        - mesh_up_features
        - mesh_down_features
        - mesh_static_features

    """

    def loads_file(fn: str) -> Any:
        """
        Load ``torch.load`` data from ``graph_dir_path``.

        Applies ``map_location`` so tensors land on the requested device.

        Parameters
        ----------
        fn : str
            The filename to load.

        Returns
        -------
        Any
            The loaded data.
        """
        return torch.load(
            os.path.join(graph_dir_path, fn),
            map_location=device,
            weights_only=True,
        )

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        [zero_index_edge_index(ei) for ei in loads_file("m2m_edge_index.pt")],
        persistent=False,
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, num_edges)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, num_edges)

    # Change first indices to 0
    # m2g and g2m has to be handled specially as not all mesh nodes
    # might be indexed
    m2g_min_indices = m2g_edge_index.min(dim=1, keepdim=True)[0]
    mesh_first = m2g_min_indices[0] < m2g_min_indices[1]
    g2m_edge_index = zero_index_g2m(
        g2m_edge_index, mesh_static_features, mesh_first=mesh_first
    )
    m2g_edge_index = zero_index_m2g(
        m2g_edge_index, mesh_static_features, mesh_first=mesh_first
    )

    assert m2g_edge_index.min() >= 0, "Negative node index in m2g"
    assert g2m_edge_index.min() >= 0, "Negative node index in g2m"

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Not just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], input_dim)
    m2m_features = loads_file("m2m_features.pt")
    g2m_features = loads_file("g2m_features.pt")  # (num_edges, input_dim)
    m2g_features = loads_file("m2g_features.pt")  # (num_edges, input_dim)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length

    m2m_features = BufferList(m2m_features, persistent=False)
    m2m_features /= longest_edge
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            [
                zero_index_edge_index(ei)
                for ei in loads_file("mesh_up_edge_index.pt")
            ],
            persistent=False,
        )  # List of (2, num_edges[l])
        mesh_down_edge_index = BufferList(
            [
                zero_index_edge_index(ei)
                for ei in loads_file("mesh_down_edge_index.pt")
            ],
            persistent=False,
        )  # List of (2, num_edges[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (num_edges[l], input_dim)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (num_edges[l], input_dim)

        # Rescale
        mesh_up_features = BufferList(mesh_up_features, persistent=False)
        mesh_up_features /= longest_edge
        mesh_down_features = BufferList(mesh_down_features, persistent=False)
        mesh_down_features /= longest_edge

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        mesh_up_edge_index = BufferList([], persistent=False)
        mesh_down_edge_index = BufferList([], persistent=False)
        mesh_up_features = BufferList([], persistent=False)
        mesh_down_features = BufferList([], persistent=False)

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }
