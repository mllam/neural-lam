# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch_geometric as pyg
import zarr
from graphcast import graphcast as gc_gc
from graphcast import grid_mesh_connectivity as gc_gm
from graphcast import icosahedral_mesh as gc_im
from graphcast import model_utils as gc_mu

GC_SPATIAL_FEATURES_KWARGS = {
    "add_node_positions": False,
    "add_node_latitude": True,
    "add_node_longitude": True,
    "add_relative_positions": True,
    "relative_longitude_local_coordinates": True,
    "relative_latitude_local_coordinates": True,
}


def vertice_cart_to_lat_lon(vertices):
    """
    Convert vertice positions to lat-lon

    vertices: (N_vert, 3), cartesian coordinates
    Returns: (N_vert, 2), lat-lon coordinates
    """
    phi, theta = gc_mu.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2]
    )
    (
        nodes_lat,
        nodes_lon,
    ) = gc_mu.spherical_to_lat_lon(phi=phi, theta=theta)
    return np.stack((nodes_lat, nodes_lon), axis=1)  # (N, 2)


def plot_graph(edge_index, pos_lat_lon, title=None):
    """
    Plot flattened global graph
    """
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos_lat_lon.shape[0])
        .cpu()
        .numpy()
    )
    edge_index = edge_index.cpu().numpy()
    # Make lon x-axis
    pos = torch.stack((pos_lat_lon[:, 1], pos_lat_lon[:, 0]), dim=1)
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis


def inter_mesh_connection(from_mesh, to_mesh):
    """
    Connect from_mesh to to_mesh
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


def main():
    """
    Global graph generation
    """
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="global_example_era5",
        help="Dataset to load grid point coordinates from "
        "(default: global_example_era5)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="global_multiscale",
        help="Name to save graph as (default: global_multiscale)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If graphs should be plotted during generation "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--splits",
        default=3,
        type=int,
        help="Number of splits to triangular mesh (default: 3)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Number of levels to keep, from finest upwards "
        "(default: None (keep all))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=0,
        help="Generate hierarchical mesh graph (default: 0, no)",
    )
    args = parser.parse_args()

    fields_group_path = os.path.join("data", args.dataset, "fields.zarr")
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)

    # Load grid positions
    fields_group = zarr.open(fields_group_path, mode="r")
    grid_lat = np.array(
        fields_group["latitude"], dtype=np.float32
    )  # (num_lat,)
    grid_lon = np.array(
        fields_group["longitude"], dtype=np.float32
    )  # (num_long,)

    # Create lat-lon grid
    grid_lat_lon = np.stack(
        (
            np.expand_dims(grid_lat, 0).repeat(
                len(grid_lon), 0
            ),  # (num_lat, num_long)
            np.expand_dims(grid_lon, 1).repeat(
                len(grid_lat), 1
            ),  # (num_lon, num_lat)
        ),
        axis=2,
    )  # (num_lon, num_lat, 2)
    num_lon, num_lat, _ = grid_lat_lon.shape
    grid_lat_lon_flat = grid_lat_lon.reshape(-1, 2)
    num_grid_nodes = grid_lat_lon_flat.shape[0]
    # flattened, (num_grid_nodes, 2)

    grid_lat_lon_torch = torch.tensor(grid_lat_lon_flat, dtype=torch.float32)
    # Save in graph dir?
    torch.save(
        grid_lat_lon_torch, os.path.join(graph_dir_path, "grid_lat_lon.pt")
    )

    # Mesh, index 0 is initial graph, with longest edges
    mesh_list = gc_im.get_hierarchy_of_triangular_meshes_for_sphere(args.splits)
    if args.levels is not None:
        assert (
            args.levels <= args.splits + 1
        ), f"Can not keep {args.levels} levels when doing {args.splits} splits"
        mesh_list = mesh_list[-args.levels :]

    if args.hierarchical:
        mesh_list_rev = list(reversed(mesh_list))  # 0 is finest graph now
        m2m_graphs = mesh_list_rev  # list of num_splitgraphs

        # Up and down edges for hierarchy
        # Reuse code for connecting grid to mesh?
        mesh_up_ei_list = []
        mesh_down_ei_list = []
        mesh_up_features_list = []
        mesh_down_features_list = []
        for from_mesh, to_mesh in zip(mesh_list_rev[:-1], mesh_list_rev[1:]):
            mesh_up_ei = inter_mesh_connection(from_mesh, to_mesh)
            # Down is opposite direction of up
            mesh_down_ei = np.stack(
                (mesh_up_ei[1, :], mesh_up_ei[0, :]), axis=0
            )
            mesh_up_ei_list.append(torch.tensor(mesh_up_ei, dtype=torch.long))
            mesh_down_ei_list.append(
                torch.tensor(mesh_down_ei, dtype=torch.long)
            )

            from_mesh_lat_lon = vertice_cart_to_lat_lon(
                from_mesh.vertices
            )  # (N, 2)
            to_mesh_lat_lon = vertice_cart_to_lat_lon(
                to_mesh.vertices
            )  # (N, 2)

            # Extract features for hierarchical edges
            _, _, mesh_up_features = gc_mu.get_bipartite_graph_spatial_features(
                senders_node_lat=from_mesh_lat_lon[:, 0],
                senders_node_lon=from_mesh_lat_lon[:, 1],
                senders=mesh_up_ei[0, :],
                receivers_node_lat=to_mesh_lat_lon[:, 0],
                receivers_node_lon=to_mesh_lat_lon[:, 1],
                receivers=mesh_up_ei[1, :],
                **GC_SPATIAL_FEATURES_KWARGS,
            )
            _, _, mesh_down_features = (
                gc_mu.get_bipartite_graph_spatial_features(
                    senders_node_lat=to_mesh_lat_lon[:, 0],
                    senders_node_lon=to_mesh_lat_lon[:, 1],
                    senders=mesh_down_ei[0, :],
                    receivers_node_lat=from_mesh_lat_lon[:, 0],
                    receivers_node_lon=from_mesh_lat_lon[:, 1],
                    receivers=mesh_down_ei[1, :],
                    **GC_SPATIAL_FEATURES_KWARGS,
                )
            )
            mesh_up_features_list.append(
                torch.tensor(mesh_up_features, dtype=torch.float32)
            )
            mesh_down_features_list.append(
                torch.tensor(mesh_down_features, dtype=torch.float32)
            )

        # Save up+down edge index + features to disk
        torch.save(
            mesh_up_ei_list,
            os.path.join(graph_dir_path, "mesh_up_edge_index.pt"),
        )
        torch.save(
            mesh_down_ei_list,
            os.path.join(graph_dir_path, "mesh_down_edge_index.pt"),
        )
        torch.save(
            mesh_up_features_list,
            os.path.join(graph_dir_path, "mesh_up_features.pt"),
        )
        torch.save(
            mesh_down_features_list,
            os.path.join(graph_dir_path, "mesh_down_features.pt"),
        )
    else:
        # Merge meshes
        # Modify gc code, as this uses some python 3.10 things
        for mesh_i, mesh_ip1 in zip(mesh_list[:-1], mesh_list[1:]):
            # itertools.pairwise(mesh_list):
            num_nodes_mesh_i = mesh_i.vertices.shape[0]
            assert np.allclose(
                mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i]
            )

        merged_mesh = gc_im.TriangularMesh(
            vertices=mesh_list[-1].vertices,
            faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
        )

        m2m_graphs = [merged_mesh]  # Should be list of len 1

    m2m_edge_index_list = []
    m2m_features_list = []
    mesh_features_list = []
    mesh_lat_lon_list = []
    for mesh_graph in m2m_graphs:
        mesh_edge_index = np.stack(
            gc_im.faces_to_edges(mesh_graph.faces), axis=0
        )
        m2m_edge_index_list.append(mesh_edge_index)

        # Compute features
        mesh_lat_lon = vertice_cart_to_lat_lon(mesh_graph.vertices)  # (N, 2)
        mesh_features, m2m_features = gc_mu.get_graph_spatial_features(
            node_lat=mesh_lat_lon[:, 0],
            node_lon=mesh_lat_lon[:, 1],
            senders=mesh_edge_index[0, :],
            receivers=mesh_edge_index[1, :],
            **GC_SPATIAL_FEATURES_KWARGS,
        )
        mesh_features_list.append(mesh_features)
        m2m_features_list.append(m2m_features)
        mesh_lat_lon_list.append(mesh_lat_lon)

        # Check that indexing is correct
        _, mesh_theta = gc_mu.lat_lon_deg_to_spherical(
            mesh_lat_lon[:, 0],
            mesh_lat_lon[:, 1],
        )
        assert np.sum(np.abs(mesh_features[:, 0] - np.cos(mesh_theta))) <= 1e-10

    # Convert to torch
    m2m_edge_index_torch = [
        torch.tensor(mesh_ei, dtype=torch.long)
        for mesh_ei in m2m_edge_index_list
    ]
    m2m_features_torch = [
        torch.tensor(m2m_features, dtype=torch.float32)
        for m2m_features in m2m_features_list
    ]
    mesh_features_torch = [
        torch.tensor(mesh_features, dtype=torch.float32)
        for mesh_features in mesh_features_list
    ]
    mesh_lat_lon_torch = [
        torch.tensor(mesh_lat_lon, dtype=torch.float32)
        for mesh_lat_lon in mesh_lat_lon_list
    ]
    # Save to static dir
    torch.save(
        m2m_edge_index_torch,
        os.path.join(graph_dir_path, "m2m_edge_index.pt"),
    )
    torch.save(
        m2m_features_torch,
        os.path.join(graph_dir_path, "m2m_features.pt"),
    )
    torch.save(
        mesh_features_torch,
        os.path.join(graph_dir_path, "mesh_features.pt"),
    )
    torch.save(
        mesh_lat_lon_torch,
        os.path.join(graph_dir_path, "mesh_lat_lon.pt"),
    )

    if args.plot:
        for level_i, (m2m_edge_index, mesh_lat_lon) in enumerate(
            zip(m2m_edge_index_torch, mesh_lat_lon_torch)
        ):
            plot_graph(
                m2m_edge_index, mesh_lat_lon, title=f"Mesh level {level_i}"
            )
            plt.show()

    # Because GC code returns indexes into flattened lat-lon matrix, we have to
    # re-map grid indices. We always work with lon-lat order, to be consistent
    # with WB2 data.
    # This creates the correct mapping for the grid indices
    grid_index_map = (
        torch.arange(num_grid_nodes).reshape(num_lon, num_lat).T.flatten()
    )

    # Grid2Mesh: Radius-based
    grid_con_mesh = m2m_graphs[0]  # Mesh graph that should be connected to grid
    grid_con_mesh_lat_lon = mesh_lat_lon_list[0]

    # Compute maximum edge distance in finest mesh
    # pylint: disable-next=protected-access
    max_mesh_edge_len = gc_gc._get_max_edge_distance(mesh_list[-1])
    g2m_connect_radius = 0.6 * max_mesh_edge_len
    g2m_grid_mesh_indices = gc_gm.radius_query_indices(
        grid_latitude=grid_lat,
        grid_longitude=grid_lon,
        mesh=grid_con_mesh,
        radius=g2m_connect_radius,
    )
    # Returns two arrays of node indices, each [num_edges]

    g2m_edge_index = np.stack(g2m_grid_mesh_indices, axis=0)
    g2m_edge_index_torch = torch.tensor(g2m_edge_index, dtype=torch.long)
    # Grid index fix
    g2m_edge_index_torch[0] = grid_index_map[g2m_edge_index_torch[0]]

    # Only care about edge features here
    _, _, g2m_features = gc_mu.get_bipartite_graph_spatial_features(
        senders_node_lat=grid_lat_lon_flat[:, 0],
        senders_node_lon=grid_lat_lon_flat[:, 1],
        senders=g2m_edge_index[0, :],
        receivers_node_lat=grid_con_mesh_lat_lon[:, 0],
        receivers_node_lon=grid_con_mesh_lat_lon[:, 1],
        receivers=g2m_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    g2m_features_torch = torch.tensor(g2m_features, dtype=torch.float32)

    torch.save(
        g2m_edge_index_torch,
        os.path.join(graph_dir_path, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_features_torch,
        os.path.join(graph_dir_path, "g2m_features.pt"),
    )

    # Mesh2Grid: Connect to containing mesh triangle
    m2g_grid_mesh_indices = gc_gm.in_mesh_triangle_indices(
        grid_latitude=grid_lat,
        grid_longitude=grid_lon,
        mesh=mesh_list[-1],
    )  # Note: Still returned in order (grid, mesh), need to inverse
    m2g_edge_index = np.stack(m2g_grid_mesh_indices[::-1], axis=0)
    m2g_edge_index_torch = torch.tensor(m2g_edge_index, dtype=torch.long)
    # Grid index fix
    m2g_edge_index_torch[1] = grid_index_map[m2g_edge_index_torch[1]]

    # Only care about edge features here
    _, _, m2g_features = gc_mu.get_bipartite_graph_spatial_features(
        senders_node_lat=grid_con_mesh_lat_lon[:, 0],
        senders_node_lon=grid_con_mesh_lat_lon[:, 1],
        senders=m2g_edge_index[0, :],
        receivers_node_lat=grid_lat_lon_flat[:, 0],
        receivers_node_lon=grid_lat_lon_flat[:, 1],
        receivers=m2g_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    m2g_features_torch = torch.tensor(m2g_features, dtype=torch.float32)

    torch.save(
        m2g_edge_index_torch,
        os.path.join(graph_dir_path, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_features_torch,
        os.path.join(graph_dir_path, "m2g_features.pt"),
    )

    num_mesh_nodes = grid_con_mesh_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
