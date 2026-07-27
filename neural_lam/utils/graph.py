"""Loading and zero-indexing of mesh/grid graph tensors."""

# Standard library
import os
import warnings
from pathlib import Path
from typing import Any, Union

# Third-party
import torch
import yaml

# Local
from .buffer_list import BufferList

LEGACY_GRAPH_SPEC_VERSION = "legacy"


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
    graph_dir_path: Union[str, Path],
    mesh_node_features_scaling: float,
    device: str = "cpu",
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
    mesh_node_features_scaling : float
        Scalar used to normalize mesh node coordinate features for graphs in
        the current on-disk format.
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

    # TODO: move graph creation/loading/versioning into its own submodule.
    # Local
    from ..create_graph import (  # Local import avoids circular imports.
        CURRENT_GRAPH_SPEC_VERSION,
        METAINFO_FILENAME,
    )

    def load_graph_spec_version() -> str:
        """
        Return the graph spec version for the graph at ``graph_dir_path`` by
        reading the ``METAINFO_FILENAME`` file and extracting the
        ``spec_version`` entry.

        Returns
        -------
        str
            The graph spec version, or ``LEGACY_GRAPH_SPEC_VERSION`` if the
            metainfo file is missing (with a warning).
        """
        metainfo_path = Path(graph_dir_path) / METAINFO_FILENAME
        if not metainfo_path.exists():
            warnings.warn(
                "Graph metainfo file is missing; assuming this graph uses "
                "the legacy pre-spec format. Mesh node feature normalization "
                "will be skipped because legacy mesh node features are assumed "
                "to already be normalized. Edge indices will be zero-offset on "
                "load to convert legacy offset node labels to the per-node-set "
                "zero-based index spaces required by the current graph spec.",
                RuntimeWarning,
                stacklevel=2,
            )
            return LEGACY_GRAPH_SPEC_VERSION

        try:
            meta = yaml.safe_load(metainfo_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Failed to parse {METAINFO_FILENAME}: {exc}"
            ) from exc

        spec_version = None if meta is None else meta.get("spec_version")
        if spec_version is None:
            raise ValueError(
                f"{METAINFO_FILENAME} is missing 'spec_version' entry"
            )
        return spec_version

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    graph_spec_version = load_graph_spec_version()
    if graph_spec_version not in {
        LEGACY_GRAPH_SPEC_VERSION,
        CURRENT_GRAPH_SPEC_VERSION,
    }:
        raise ValueError(
            "Unsupported graph spec version "
            f"{graph_spec_version!r} in {METAINFO_FILENAME}"
        )

    should_normalize_mesh_features = (
        graph_spec_version == CURRENT_GRAPH_SPEC_VERSION
    )
    should_zero_index_edge_indices = (
        graph_spec_version == LEGACY_GRAPH_SPEC_VERSION
    )

    # Normalize static mesh features for the current on-disk graph format.
    # Legacy graphs already store normalized mesh coordinates.
    if should_normalize_mesh_features:
        if mesh_node_features_scaling == 0:
            warnings.warn(
                "Mesh node feature scaling is zero; falling back to 1.0 so "
                "mesh node coordinates are left unchanged after graph "
                "loading.",
                RuntimeWarning,
                stacklevel=2,
            )
            mesh_node_features_scaling = 1.0

        for m in mesh_static_features:
            m[:, :2] /= mesh_node_features_scaling

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, num_edges)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, num_edges)

    if should_zero_index_edge_indices:
        # Legacy graphs used a shifted node-index layout; normalize it on load.
        m2m_edge_index = BufferList(
            [zero_index_edge_index(ei) for ei in m2m_edge_index],
            persistent=False,
        )

        # m2g and g2m has to be handled specially as not all mesh nodes
        # might be indexed.
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
        mesh_up_edge_index_aslist = loads_file("mesh_up_edge_index.pt")
        mesh_down_edge_index_aslist = loads_file("mesh_down_edge_index.pt")

        # Legacy graphs used a shifted node-index layout; but internally in
        # neural-lam we expect edge indices to be zero-indexed within their
        # respective node sets, so we need to zero-index these edges indexes.
        if should_zero_index_edge_indices:
            mesh_up_edge_index_aslist = [
                zero_index_edge_index(ei) for ei in mesh_up_edge_index_aslist
            ]
            mesh_down_edge_index_aslist = [
                zero_index_edge_index(ei) for ei in mesh_down_edge_index_aslist
            ]

        mesh_up_edge_index = BufferList(
            mesh_up_edge_index_aslist, persistent=False
        )
        mesh_down_edge_index = BufferList(
            mesh_down_edge_index_aslist, persistent=False
        )

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
