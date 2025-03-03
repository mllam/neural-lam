# Third-party
import numpy as np
import scipy
import torch
from graphcast import graphcast as gc_gc
from graphcast import icosahedral_mesh as gc_im
from graphcast import model_utils as gc_mu

# First-party
import neural_lam.graphs.graph_utils as gutils

# Keyword arguments to use when calling graphcast functions
# for creating graph features
GC_SPATIAL_FEATURES_KWARGS = {
    "add_node_positions": False,
    "add_node_latitude": True,
    "add_node_longitude": True,
    "add_relative_positions": True,
    "relative_longitude_local_coordinates": True,
    "relative_latitude_local_coordinates": True,
}


def inter_mesh_connection(from_mesh, to_mesh):
    """Connect finer from_mesh to coarser to_mesh.

    Parameters
    ----------
    from_mesh : trimesh.Trimesh
        The mesh to connect from.
    to_mesh : trimesh.Trimesh
        The mesh to connect to.

    Returns
    -------
    np.array
        Edge index describing connections, shaped (2, num_edges).
    """
    kd_tree = scipy.spatial.cKDTree(to_mesh.vertices)

    # Each node on lower (from) mesh will connect to 1 or 2 on level above
    # pylint: disable-next=protected-access
    radius = 1.1 * gc_gc._get_max_edge_distance(from_mesh)
    query_indices = kd_tree.query_ball_point(x=from_mesh.vertices, r=radius)

    from_edge_indices = []
    to_edge_indices = []
    for from_index, to_neighbors in enumerate(query_indices):
        from_edge_indices.append(np.repeat(from_index, len(to_neighbors)))
        to_edge_indices.append(to_neighbors)

    from_edge_indices = np.concatenate(from_edge_indices, axis=0).astype(int)
    to_edge_indices = np.concatenate(to_edge_indices, axis=0).astype(int)

    edge_index = np.stack(
        (from_edge_indices, to_edge_indices), axis=0
    )  # (2, M)
    return edge_index


def _create_mesh_levels(splits, levels=None):
    """Create a sequence of mesh graph levels by splitting a global icosahedron.

    Parameters
    ----------
    splits : int
        Number of times to split icosahedron.
    levels : int, optional
        Number of levels to keep (from finest resolution and up).
        If None, keep all levels.

    Returns
    -------
    list : List[trimesh.Trimesh]
        List of mesh levels.
    """
    # Mesh, index 0 is initial graph, with longest edges
    mesh_list = gc_im.get_hierarchy_of_triangular_meshes_for_sphere(splits)
    if levels is not None:
        assert (
            levels <= splits + 1
        ), f"Can not keep {levels} levels when doing {splits} splits"
        mesh_list = mesh_list[-levels:]

    return mesh_list


def create_multiscale_mesh(splits, levels):
    """Create a multiscale triangular mesh graph.

    Parameters
    ----------
    splits : int
        Number of times to split icosahedron.
    levels : int
        Number of levels to keep (from finest resolution and up).

    Returns
    -------
    graphcast.icosahedral_mesh.TriangularMesh
        The merged mesh.
    list : List[trimesh.Trimesh]
        List of individual mesh levels.
    """
    mesh_list = _create_mesh_levels(splits, levels)

    # Merge meshes
    # Modified gc code, as it uses some python 3.10 things
    for mesh_i, mesh_ip1 in zip(mesh_list[:-1], mesh_list[1:]):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(
            mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i]
        )

    merged_mesh = gc_im.TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )

    return merged_mesh, mesh_list


def create_hierarchical_mesh(splits, levels, crop_chull=None):
    """Create a hierarchical triangular mesh graph.

    Parameters
    ----------
    splits : int
        Number of times to split icosahedron.
    levels : int
        Number of levels to keep (from finest resolution and up).
    crop_chull : spherical_geometry.SphericalPolygon, optional
        A convex hull to crop graphs to within. If None no cropping is done.

    Returns
    -------
    List[trimesh.Trimesh]
        Levels in hierarchical graph.
    List[torch.Tensor]
        List of edge index for upwards inter-level edges,
        each of shape (2, num_up_edges).
    List[torch.Tensor]
        List of edge index for downwards inter-level edges,
        each of shape (2, num_down_edges).
    List[torch.Tensor]
        List of edge features for up edges,
        each of shape (num_up_edges, d_edge_features).
    List[torch.Tensor]
        List of edge features for down edges,
        each of shape (num_down_edges, d_edge_features).
    """
    mesh_list = _create_mesh_levels(splits, levels)

    m2m_graphs = list(reversed(mesh_list))  # 0 is finest graph now

    if crop_chull is not None:
        # Crop m2m graphs here, and used cropped versions below
        m2m_graphs = [
            gutils.subset_mesh_to_chull(crop_chull, mesh) for mesh in m2m_graphs
        ]

    # Up and down edges for hierarchy
    # Reuse code for connecting grid to mesh?
    mesh_up_ei_list = []
    mesh_down_ei_list = []
    mesh_up_features_list = []
    mesh_down_features_list = []
    for from_mesh, to_mesh in zip(m2m_graphs[:-1], m2m_graphs[1:]):
        # Compute up and down inter-mesh edges
        mesh_up_ei = inter_mesh_connection(from_mesh, to_mesh)
        # Down edges have opposite direction of up
        mesh_down_ei = np.stack((mesh_up_ei[1, :], mesh_up_ei[0, :]), axis=0)

        mesh_up_ei_list.append(torch.tensor(mesh_up_ei, dtype=torch.long))
        mesh_down_ei_list.append(torch.tensor(mesh_down_ei, dtype=torch.long))

        # Compute features for inter-mesh edges
        mesh_up_features = create_edge_features(
            mesh_up_ei, sender_mesh=from_mesh, receiver_mesh=to_mesh
        )
        mesh_down_features = create_edge_features(
            mesh_down_ei, sender_mesh=to_mesh, receiver_mesh=from_mesh
        )
        mesh_up_features_list.append(mesh_up_features)
        mesh_down_features_list.append(mesh_down_features)

    return (
        m2m_graphs,
        mesh_up_ei_list,
        mesh_down_ei_list,
        mesh_up_features_list,
        mesh_down_features_list,
    )


def connect_to_mesh_radius(grid_pos, mesh: gc_im.TriangularMesh, radius: float):
    """Create edge_index that connects given grid positions to mesh, if
    within specific radius of mesh node.

    Parameters
    ----------
    grid_pos : np.array
        (num_grid_nodes, 2) array containing lat-lons of grid nodes.
    mesh : trimesh.Trimesh
        The mesh to connect to.
    radius : float
        The radius to connect within (in euclidean distance).

    Returns
    -------
    torch.Tensor
        Edge index tensor connecting grid to mesh nodes.
    """
    grid_mesh_indices = gutils.radius_query_indices_irregular(
        grid_lat_lon=grid_pos,
        mesh=mesh,
        radius=radius,
    )
    # Returns two arrays of node indices, each [num_edges]

    # Stacking order to have from grid to mesh
    edge_index = np.stack(grid_mesh_indices, axis=0)
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)
    return edge_index_torch


def connect_to_grid_containing_tri(grid_pos, mesh: gc_im.TriangularMesh):
    """Create edge_index by for each grid node finding the containing triangle
    in the mesh and creating edges from the corner nodes to the grid node.

    Parameters
    ----------
    grid_pos : np.array
        (num_grid_nodes, 2) array containing lat-lons of grid nodes.
    mesh : trimesh.Trimesh
        The mesh to connect from.

    Returns
    -------
    torch.Tensor
        Edge index tensor connecting mesh to grid nodes.
    """
    grid_mesh_indices = gutils.in_mesh_triangle_indices_irregular(
        grid_lat_lon=grid_pos,
        mesh=mesh,
    )

    # Note: Still returned in order (grid, mesh), need to inverse
    edge_index = np.stack(grid_mesh_indices[::-1], axis=0)

    # Make torch tensor
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)
    return edge_index_torch


def create_mesh_graph_features(mesh_graph: gc_im.TriangularMesh):
    """Create torch tensors for edge_index and features
    from single TriangularMesh.

    Parameters
    ----------
    mesh_graph : trimesh.Trimesh
        The triangular mesh graph to extract features from.

    Returns
    -------
    torch.Tensor
        Edge index tensor of shape (2, num_edges).
    torch.Tensor
        Node features tensor of shape (num_nodes, d_node_features).
    torch.Tensor
        Edge features tensor of shape (num_edges, d_edge_features).
    torch.Tensor
        Node positions in lat-lon, shape (num_nodes, 2).
    """
    mesh_edge_index = np.stack(gc_im.faces_to_edges(mesh_graph.faces), axis=0)

    # Compute features
    mesh_lat_lon = gutils.node_cart_to_lat_lon(mesh_graph.vertices)  # (N, 2)
    mesh_node_features, mesh_edge_features = gc_mu.get_graph_spatial_features(
        node_lat=mesh_lat_lon[:, 1],
        node_lon=mesh_lat_lon[:, 0],
        senders=mesh_edge_index[0, :],
        receivers=mesh_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )

    return (
        torch.tensor(mesh_edge_index, dtype=torch.long),
        torch.tensor(mesh_node_features, dtype=torch.float32),
        torch.tensor(mesh_edge_features, dtype=torch.float32),
        torch.tensor(mesh_lat_lon, dtype=torch.float32),
    )


def create_edge_features(
    edge_index,
    sender_coords=None,
    receiver_coords=None,
    sender_mesh=None,
    receiver_mesh=None,
):
    """Create torch tensors with edge features for given edge_index.
    For sender and receiver, either coords or a mesh has to be given.

    Parameters
    ----------
    edge_index : np.array
        Edge index array of shape (2, num_edges).
    sender_coords : np.array, optional
        Coordinates of sender nodes, shape (num_sender_nodes, 2).
    receiver_coords : np.array, optional
        Coordinates of receiver nodes, shape (num_receiver_nodes, 2).
    sender_mesh : trimesh.Trimesh, optional
        Mesh containing sender nodes.
    receiver_mesh : trimesh.Trimesh, optional
        Mesh containing receiver nodes.

    Returns
    -------
    torch.Tensor
        Edge features tensor of shape (num_edges, d_edge_features).
    """
    if sender_mesh is not None:
        assert (
            sender_coords is None
        ), "Can not extract features using both sender coords and sender mesh"
        sender_coords = gutils.node_cart_to_lat_lon(sender_mesh.vertices)
        # (N, 2)
    assert sender_coords is not None, (
        "Either sender_coords or sender_mesh has to be given to "
        "create_edge_features"
    )

    if receiver_mesh is not None:
        assert receiver_coords is None, (
            "Can not extract features using both receiver coords and "
            "receiver mesh"
        )
        receiver_coords = gutils.node_cart_to_lat_lon(receiver_mesh.vertices)
        # (N, 2)
    assert receiver_coords is not None, (
        "Either receiver_coords or receiver_mesh has to be given to "
        "create_edge_features"
    )

    # Make sure all coords have same dtype
    sender_coords = sender_coords.astype(np.float32)
    receiver_coords = receiver_coords.astype(np.float32)

    _, _, edge_features = gc_mu.get_bipartite_graph_spatial_features(
        senders_node_lat=sender_coords[:, 0],
        senders_node_lon=sender_coords[:, 1],
        senders=edge_index[0, :],
        receivers_node_lat=receiver_coords[:, 0],
        receivers_node_lon=receiver_coords[:, 1],
        receivers=edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    return torch.tensor(edge_features, dtype=torch.float32)
