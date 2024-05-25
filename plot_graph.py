# Standard library
from argparse import ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch_geometric as pyg

# First-party
from neural_lam import config, utils

MESH_HEIGHT = 0.1
MESH_LEVEL_DIST = 0.2
GRID_HEIGHT = 0


def main():
    """
    Plot graph structure in 3D using plotly
    """
    parser = ArgumentParser(description="Plot graph")
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to plot (default: multiscale)",
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
    config_loader = config.Config.from_file(args.data_config)

    # Load graph data
    hierarchical, graph_ldict = utils.load_graph(args.graph)
    (g2m_edge_index, m2g_edge_index, m2m_edge_index,) = (
        graph_ldict["g2m_edge_index"],
        graph_ldict["m2g_edge_index"],
        graph_ldict["m2m_edge_index"],
    )
    mesh_up_edge_index, mesh_down_edge_index = (
        graph_ldict["mesh_up_edge_index"],
        graph_ldict["mesh_down_edge_index"],
    )
    mesh_static_features = graph_ldict["mesh_static_features"]

    grid_static_features = utils.load_static_data(config_loader.dataset.name)[
        "grid_static_features"
    ]

    # Extract values needed, turn to numpy
    grid_pos = grid_static_features[:, :2].numpy()
    # Add in z-dimension
    z_grid = GRID_HEIGHT * np.ones((grid_pos.shape[0],))
    grid_pos = np.concatenate(
        (grid_pos, np.expand_dims(z_grid, axis=1)), axis=1
    )

    # List of edges to plot, (edge_index, color, line_width, label)
    edge_plot_list = [
        (m2g_edge_index.numpy(), "black", 0.4, "M2G"),
        (g2m_edge_index.numpy(), "black", 0.4, "G2M"),
    ]

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        mesh_level_pos = [
            np.concatenate(
                (
                    level_static_features.numpy(),
                    MESH_HEIGHT
                    + MESH_LEVEL_DIST
                    * height_level
                    * np.ones((level_static_features.shape[0], 1)),
                ),
                axis=1,
            )
            for height_level, level_static_features in enumerate(
                mesh_static_features, start=1
            )
        ]
        mesh_pos = np.concatenate(mesh_level_pos, axis=0)

        # Add inter-level mesh edges
        edge_plot_list += [
            (level_ei.numpy(), "blue", 1, f"M2M Level {level}")
            for level, level_ei in enumerate(m2m_edge_index)
        ]

        # Add intra-level mesh edges
        up_edges_ei = np.concatenate(
            [level_up_ei.numpy() for level_up_ei in mesh_up_edge_index], axis=1
        )
        down_edges_ei = np.concatenate(
            [level_down_ei.numpy() for level_down_ei in mesh_down_edge_index],
            axis=1,
        )
        edge_plot_list.append((up_edges_ei, "green", 1, "Mesh up"))
        edge_plot_list.append((down_edges_ei, "green", 1, "Mesh down"))

        mesh_node_size = 2.5
    else:
        mesh_pos = mesh_static_features.numpy()

        mesh_degrees = pyg.utils.degree(m2m_edge_index[1]).numpy()
        z_mesh = MESH_HEIGHT + 0.01 * mesh_degrees
        mesh_node_size = mesh_degrees / 2

        mesh_pos = np.concatenate(
            (mesh_pos, np.expand_dims(z_mesh, axis=1)), axis=1
        )

        edge_plot_list.append((m2m_edge_index.numpy(), "blue", 1, "M2M"))

    # All node positions in one array
    node_pos = np.concatenate((mesh_pos, grid_pos), axis=0)

    # Add edges
    data_objs = []
    for (
        ei,
        col,
        width,
        label,
    ) in edge_plot_list:
        edge_start = node_pos[ei[0]]  # (M, 2)
        edge_end = node_pos[ei[1]]  # (M, 2)
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

        scatter_obj = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line={"color": col, "width": width},
            name=label,
        )
        data_objs.append(scatter_obj)

    # Add node objects

    data_objs.append(
        go.Scatter3d(
            x=grid_pos[:, 0],
            y=grid_pos[:, 1],
            z=grid_pos[:, 2],
            mode="markers",
            marker={"color": "black", "size": 1},
            name="Grid nodes",
        )
    )
    data_objs.append(
        go.Scatter3d(
            x=mesh_pos[:, 0],
            y=mesh_pos[:, 1],
            z=mesh_pos[:, 2],
            mode="markers",
            marker={"color": "blue", "size": mesh_node_size},
            name="Mesh nodes",
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
