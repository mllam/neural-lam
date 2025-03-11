# Standard library
import argparse
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
from graphcast import graphcast as gc_gc
from spherical_geometry.polygon import SphericalPolygon

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import create as gcreate
from .graphs import graph_utils as gutils
from .graphs import vis as gvis


def main():
    parser = argparse.ArgumentParser(
        description="Triangular graph generation using weather-models-graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs and outputs
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation ",
    )

    # Graph structure
    parser.add_argument(
        "--splits",
        default=3,
        type=int,
        help="Number of splits to triangular mesh",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Number of levels to keep, from finest upwards ",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        help="If the graph should be global, not cropped based on grid nodes",
    )
    parser.add_argument(
        "--rotate_ico",
        action="store_true",
        help="If the original icosahedron should be rotated to better line "
        "up with the grid data",
    )
    parser.add_argument(
        "--g2m_radius",
        type=float,
        help="Radius (relative to longest mesh edge) to connect grid and "
        "mesh nodes in g2m.",
        default=0.6,
    )
    parser.add_argument(
        "--g2m_radius_boundary",
        type=float,
        help="Radius (relative to longest mesh edge) to connect boundary "
        "grid nodes and mesh nodes in g2m, when config has boundary forcing.",
        default=1.8,
    )
    parser.add_argument(
        "--allow_disconnected",
        action="store_true",
        help="Allow disconnected nodes in g2m. This is generally a bad idea and"
        "should only be used for testing purposes.",
    )
    args = parser.parse_args()

    _, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    # Set up dir for saving graph
    save_dir_path = os.path.join(datastore.root_path, "graphs", args.graph_name)
    os.makedirs(save_dir_path, exist_ok=True)

    # Load grid positions
    grid_lat_lon = utils.get_stacked_lat_lons(
        datastore, datastore_boundary
    ).astype(
        np.float32
    )  # Must be float32 for interoperability with gc code
    # (num_nodes_full, 2)
    num_grid_nodes = grid_lat_lon.shape[0]

    # Make all longitudes be in [0, 360]
    grid_lat_lon[:, 0] = (grid_lat_lon[:, 0] + 360.0) % 360.0

    grid_xyz = gutils.node_lat_lon_to_cart(grid_lat_lon)
    grid_lat_lon_torch = torch.tensor(grid_lat_lon, dtype=torch.float32)
    torch.save(
        grid_lat_lon_torch, os.path.join(save_dir_path, "grid_lat_lon.pt")
    )

    # Workaround for global being reserved python keyword
    global_graph = getattr(args, "global")

    boundary_region = datastore_boundary is not None
    if boundary_region:
        # LAM setting with boundary
        # Construct mask to decode only to interior
        decode_mask = utils.get_interior_mask(datastore, datastore_boundary)
        interior_lat_lon = grid_lat_lon[decode_mask]
        boundary_lat_lon = grid_lat_lon[~decode_mask]

        # Only decode to interior
        grid_decode_lat_lon = interior_lat_lon
    else:
        grid_decode_lat_lon = grid_lat_lon

    if global_graph:
        # No cropping to chull
        grid_chull = None
    if not global_graph:
        # Crop mesh graph to convex hull of grid points
        # Compute convex hull
        print("Cropping for LAM model. Computing convex hull...")
        grid_chull = SphericalPolygon.convex_hull(grid_xyz)

    if args.rotate_ico:
        # Compute a point to line up icosahedron with for later
        # Use mean of grid coords, projected to surface
        # Not strictly optimal point, but perfectly good in standard setups

        grid_xyz_mean = np.mean(grid_xyz, axis=0)
        rot_point = grid_xyz_mean / np.linalg.norm(grid_xyz_mean)
    else:
        rot_point = None

    # === Create mesh graph ===
    if args.hierarchical:
        # Build hierarchical graph
        (
            m2m_graphs,  # First index is maximum refinement
            mesh_up_ei_list,
            mesh_down_ei_list,
            mesh_up_features_list,
            mesh_down_features_list,
        ) = gcreate.create_hierarchical_mesh(
            args.splits, args.levels, grid_chull, rotate_to_point=rot_point
        )
        print("Created hierarchical graph with levels:")
        for level_i, mesh in enumerate(m2m_graphs):
            print(f" {level_i} with {mesh.vertices.shape[0]} nodes")

        # Save up+down edge index + features to disk
        torch.save(
            mesh_up_ei_list,
            os.path.join(save_dir_path, "mesh_up_edge_index.pt"),
        )
        torch.save(
            mesh_down_ei_list,
            os.path.join(save_dir_path, "mesh_down_edge_index.pt"),
        )
        torch.save(
            mesh_up_features_list,
            os.path.join(save_dir_path, "mesh_up_features.pt"),
        )
        torch.save(
            mesh_down_features_list,
            os.path.join(save_dir_path, "mesh_down_features.pt"),
        )
        max_mesh_edge_len = gc_gc._get_max_edge_distance(m2m_graphs[0])
        grid_con_mesh = m2m_graphs[0]  # Mesh that should be connected to grid
    else:
        merged_mesh, mesh_list = gcreate.create_multiscale_mesh(
            args.splits, args.levels, rotate_to_point=rot_point
        )
        max_mesh_edge_len = gc_gc._get_max_edge_distance(mesh_list[-1])

        # Must use final mesh from mesh list here, as we don't want larger
        # faces (triangles) present when constructing g2m and m2g
        grid_con_mesh = mesh_list[-1]  # Mesh that should be connected to grid

        if not global_graph:
            print("Subsetting mesh graph...")
            merged_mesh = gutils.subset_mesh_to_chull(grid_chull, merged_mesh)
            # Subset also grid_con_mesh. Works since it shares the nodes with
            # merged_mesh (although not faces).
            # Note: Could be sped up, as we are now doing the exact same
            # subsetting twice
            grid_con_mesh = gutils.subset_mesh_to_chull(
                grid_chull, grid_con_mesh
            )

        print(
            "Created multiscale graph with "
            f"{merged_mesh.vertices.shape[0]} nodes"
        )
        m2m_graphs = [merged_mesh]
    mesh_graph_features = [
        gcreate.create_mesh_graph_features(mesh_graph)
        for mesh_graph in m2m_graphs
    ]
    # Ordering: edge_index, node_features, edge_features, lat_lon

    # Save to static dir
    for feat_index, file_name in enumerate(
        (
            "m2m_edge_index.pt",
            "m2m_node_features.pt",
            "m2m_features.pt",
            "mesh_lat_lon.pt",
        )
    ):
        torch.save(
            # Save as list
            [feats[feat_index] for feats in mesh_graph_features],
            os.path.join(save_dir_path, file_name),
        )

    if args.plot:
        # Plot each mesh graph level
        for level_i, (level_edge_index, _, _, level_lat_lon) in enumerate(
            mesh_graph_features
        ):
            gvis.plot_graph(
                level_edge_index, level_lat_lon, title=f"Mesh level {level_i}"
            )
            plt.show()

    # === Grid2Mesh ===
    # Grid2Mesh: Radius-based
    grid_con_lat_lon = mesh_graph_features[0][-1]
    num_mesh_nodes = grid_con_lat_lon.shape[0]

    # Compute maximum edge distance in finest mesh
    # pylint: disable-next=protected-access
    print(
        f"Edge length at bottom mesh level: {max_mesh_edge_len} "
        f"(~{max_mesh_edge_len*6378} km)"
    )

    if boundary_region:
        # Different radius for g2m connections for interior and boundary
        interior_connect_radius = args.g2m_radius * max_mesh_edge_len
        boundary_connect_radius = args.g2m_radius_boundary * max_mesh_edge_len

        # Compute g2m from both
        interior_g2m_edge_index = gcreate.connect_to_mesh_radius(
            interior_lat_lon, grid_con_mesh, interior_connect_radius
        )  # (2, num_edges_interior)
        boundary_g2m_edge_index = gcreate.connect_to_mesh_radius(
            boundary_lat_lon, grid_con_mesh, boundary_connect_radius
        )  # (2, num_edges_boundary)
        # Note: grid node indices direct to boundary positions

        # Combine, offsetting boundary node indices
        boundary_g2m_edge_index_offset = torch.stack(
            (
                # Boundary nodes always after interior nodes
                datastore.num_grid_points + boundary_g2m_edge_index[0],
                boundary_g2m_edge_index[1],
            ),
            dim=0,
        )
        g2m_edge_index = torch.cat(
            (interior_g2m_edge_index, boundary_g2m_edge_index_offset), dim=1
        )

    else:
        # Directly connect all grid nodes to mesh
        g2m_connect_radius = args.g2m_radius * max_mesh_edge_len
        g2m_edge_index = gcreate.connect_to_mesh_radius(
            grid_lat_lon, grid_con_mesh, g2m_connect_radius
        )

    if args.plot:
        gvis.plot_graph(
            g2m_edge_index,
            from_node_pos=grid_lat_lon_torch,
            to_node_pos=grid_con_lat_lon,
            title="Grid2Mesh",
        )
        plt.show()

    # Assert that all nodes are connected in g2m
    g2m_degrees_grid = pyg.utils.degree(g2m_edge_index[0], num_grid_nodes)
    g2m_degrees_mesh = pyg.utils.degree(g2m_edge_index[1], num_mesh_nodes)
    num_disc_grid = 0
    num_disc_mesh = 0
    if g2m_degrees_grid.min() == 0:
        num_disc_grid = torch.sum(g2m_degrees_grid == 0).item()
    elif g2m_degrees_mesh.min() == 0:
        num_disc_mesh = torch.sum(g2m_degrees_mesh == 0).item()
    if not (num_disc_grid == 0 and num_disc_mesh == 0):
        print(
            (
                "Disconnected nodes in G2M: "
                f"{num_disc_grid} disconnected grid nodes, "
                f"{num_disc_mesh} disconnected mesh nodes"
            )
        )
        if not args.allow_disconnected:
            assert False, "Disconnected g2m nodes."

    # Get edge features for g2m
    g2m_edge_features = gcreate.create_edge_features(
        g2m_edge_index, sender_coords=grid_lat_lon, receiver_mesh=grid_con_mesh
    )

    # Save g2m
    torch.save(
        g2m_edge_index,
        os.path.join(save_dir_path, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_edge_features,
        os.path.join(save_dir_path, "g2m_features.pt"),
    )

    # === Mesh2Grid ===
    # Mesh2Grid: Connect from containing mesh triangle

    m2g_edge_index = gcreate.connect_to_grid_containing_tri(
        grid_decode_lat_lon, grid_con_mesh
    )

    # Get edge features for m2g
    m2g_edge_features = gcreate.create_edge_features(
        m2g_edge_index,
        receiver_coords=grid_decode_lat_lon,
        sender_mesh=grid_con_mesh,
    )

    if args.plot:
        gvis.plot_graph(
            m2g_edge_index,
            from_node_pos=grid_con_lat_lon,
            to_node_pos=torch.tensor(grid_decode_lat_lon, dtype=torch.float32),
            title="Grid2Mesh",
        )
        plt.show()

    # Save m2g
    torch.save(
        m2g_edge_index,
        os.path.join(save_dir_path, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_edge_features,
        os.path.join(save_dir_path, "m2g_features.pt"),
    )

    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes} mesh nodes"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
