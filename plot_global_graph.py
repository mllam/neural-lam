# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch
from graphcast import model_utils as gc_mu

# First-party
from neural_lam import utils

MESH_HEIGHT = 1.1
MESH_LEVEL_DIST = 0.5
GRID_HEIGHT = 1


def torch_lat_lon_to_cartesian(node_lat_lon):
    """
    Convert at torch tensor with lat-lon coordinates to cartesian coordinates

    node_lat_lon: (N, 2)

    Returns:
        node_cart: (N, 3)
    """
    node_lat_lon_np = node_lat_lon.numpy()
    phi_np, theta_np = gc_mu.lat_lon_deg_to_spherical(
        node_lat_lon_np[:, 0], node_lat_lon_np[:, 1]
    )
    cart_coords_list = gc_mu.spherical_to_cartesian(phi_np, theta_np)
    return torch.tensor(np.stack(cart_coords_list, axis=1), dtype=torch.float32)


def create_edge_plot(
    edge_index,
    from_node_lat_lon,
    to_node_lat_lon,
    label,
    color="blue",
    width=1,
    from_radius=1,
    to_radius=1,
):
    """
    Create a plotly object showing edges

    edge_index: (2, M)
    from_node_lat_lon: (N, 2), positions of sender nodes
    to_node_lat_lon: (N, 2), positions of receiver nodes
    label: str, label of plot object
    """
    from_node_cart = torch_lat_lon_to_cartesian(from_node_lat_lon) * from_radius
    to_node_cart = torch_lat_lon_to_cartesian(to_node_lat_lon) * to_radius

    edge_start = from_node_cart[edge_index[0]].numpy()  # (M, 2)
    edge_end = to_node_cart[edge_index[1]].numpy()  # (M, 2)
    n_edges = edge_start.shape[0]

    x_edges = np.stack(
        (edge_start[:, 0], edge_end[:, 0], np.full(n_edges, None)), axis=1
    ).flatten()
    y_edges = np.stack(
        (edge_start[:, 1], edge_end[:, 1], np.full(n_edges, None)), axis=1
    ).flatten()
    z_edges = np.stack(
        (edge_start[:, 2], edge_end[:, 2], np.full(n_edges, None)), axis=1
    ).flatten()

    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": color, "width": width},
        name=label,
    )


def create_node_plot(node_lat_lon, label, color="blue", size=1, radius=1):
    """
    Create a plotly object showing nodes

    node_lat_lon: (N, 2)
    label: str, label of plot object
    """
    node_pos = torch_lat_lon_to_cartesian(node_lat_lon) * radius
    return go.Scatter3d(
        x=node_pos[:, 0],
        y=node_pos[:, 1],
        z=node_pos[:, 2],
        mode="markers",
        marker={"color": color, "size": size},
        name=label,
    )


def main():
    """
    Plot graph structure in 3D using plotly
    """
    parser = ArgumentParser(description="Plot graph")
    parser.add_argument(
        "--dataset",
        type=str,
        default="global_example_era5",
        help=(
            "Datast to load grid coordinates from "
            "(default: global_example_era5)"
        ),
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="global_multiscale",
        help="Graph to plot (default: global_multiscale)",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Name of .html file to save interactive plot to (default: None)",
    )
    parser.add_argument(
        "--show_axis",
        type=int,
        default=0,
        help="If the axis should be displayed (default: 0 (No))",
    )

    args = parser.parse_args()

    # Load graph data
    hierarchical, graph_ldict = utils.load_graph(args.graph)
    (
        g2m_edge_index,
        m2g_edge_index,
        m2m_edge_index,
    ) = (
        graph_ldict["g2m_edge_index"],
        graph_ldict["m2g_edge_index"],
        graph_ldict["m2m_edge_index"],
    )
    mesh_up_edge_index, mesh_down_edge_index = (
        graph_ldict["mesh_up_edge_index"],
        graph_ldict["mesh_down_edge_index"],
    )
    mesh_lat_lon = torch.load(
        os.path.join("graphs", args.graph, "mesh_lat_lon.pt")
    )  # (num_mesh, 2)
    grid_lat_lon = torch.load(
        os.path.join("graphs", args.graph, "grid_lat_lon.pt")
    )  # (num_grid, 2)

    # Add plotting objects to this list
    data_objs = []

    # Plot G2M
    data_objs.append(
        create_node_plot(
            grid_lat_lon, "Grid Nodes", color="black", radius=GRID_HEIGHT
        )
    )
    data_objs.append(
        create_edge_plot(
            g2m_edge_index,
            grid_lat_lon,
            mesh_lat_lon[0],
            "G2M Edges",
            color="black",
            width=0.4,
            from_radius=GRID_HEIGHT,
            to_radius=MESH_HEIGHT,
        )
    )

    # Plot M2G
    data_objs.append(
        create_edge_plot(
            m2g_edge_index,
            mesh_lat_lon[0],
            grid_lat_lon,
            "M2G Edges",
            color="black",
            width=0.4,
            from_radius=MESH_HEIGHT,
            to_radius=GRID_HEIGHT,
        )
    )

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        # Plot mesh layers
        for level_i, (level_lat_lon, level_edge_index) in enumerate(
            zip(mesh_lat_lon, m2m_edge_index)
        ):
            mesh_level_radius = MESH_HEIGHT + level_i * MESH_LEVEL_DIST
            data_objs.append(
                create_node_plot(
                    level_lat_lon,
                    f"Mesh Nodes (l = {level_i})",
                    size=2.5,
                    radius=mesh_level_radius,
                )
            )
            data_objs.append(
                create_edge_plot(
                    level_edge_index,
                    level_lat_lon,
                    level_lat_lon,
                    f"Mesh Edges (l = {level_i})",
                    from_radius=mesh_level_radius,
                    to_radius=mesh_level_radius,
                )
            )

        # Plot inter-level edges
        for from_level_i, (
            from_lat_lon,
            to_lat_lon,
            up_edge_index,
            down_edge_index,
        ) in enumerate(
            zip(
                mesh_lat_lon[:-1],
                mesh_lat_lon[1:],
                mesh_up_edge_index,
                mesh_down_edge_index,
            )
        ):
            from_level_radius = MESH_HEIGHT + from_level_i * MESH_LEVEL_DIST
            to_level_radius = MESH_HEIGHT + (from_level_i + 1) * MESH_LEVEL_DIST
            data_objs.append(
                create_edge_plot(
                    up_edge_index,
                    from_lat_lon,
                    to_lat_lon,
                    f"Up Edges ({from_level_i} to {from_level_i+1})",
                    from_radius=from_level_radius,
                    to_radius=to_level_radius,
                    color="green",
                    width=1,
                )
            )
            data_objs.append(
                create_edge_plot(
                    down_edge_index,
                    to_lat_lon,
                    from_lat_lon,
                    f"Down Edges ({from_level_i+1} to {from_level_i})",
                    from_radius=to_level_radius,
                    to_radius=from_level_radius,
                    color="green",
                    width=1,
                )
            )
    else:
        data_objs.append(
            create_node_plot(
                mesh_lat_lon[0], "Mesh Nodes", size=2.5, radius=MESH_HEIGHT
            )
        )
        data_objs.append(
            create_edge_plot(
                m2m_edge_index,
                mesh_lat_lon[0],
                mesh_lat_lon[0],
                "Mesh Edges",
                from_radius=MESH_HEIGHT,
                to_radius=MESH_HEIGHT,
            )
        )

    fig = go.Figure(data=data_objs)

    fig.update_layout(scene_aspectmode="data")
    fig.update_traces(connectgaps=False)

    if not args.show_axis:
        # Hide axis
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")
    else:
        fig.show()


if __name__ == "__main__":
    main()
