# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch_geometric as pyg

# Local
from . import utils
from .config import load_config_and_datastore

MESH_HEIGHT = 0.1
MESH_LEVEL_DIST = 0.2
GRID_HEIGHT = 0


def plot_graph(
    grid_pos,
    hierarchical,
    graph_ldict,
    show_axis=False,
    save=None,
):
    """Build a 3D plotly figure of the graph structure.

    Parameters
    ----------
    grid_pos : np.ndarray
        Grid node positions, shape (N_grid, 2).
    hierarchical : bool
        Whether the loaded graph is hierarchical.
    graph_ldict : dict
        Graph dict as returned by ``utils.load_graph``.
    show_axis : bool
        If True, show the 3D axis.
    save : str or None
        If given, save the figure as an HTML file at this path.

    Returns
    -------
    go.Figure
        The plotly figure object.
    """
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
    mesh_static_features = graph_ldict["mesh_static_features"]

    # Add in z-dimension
    z_grid = GRID_HEIGHT * np.ones((grid_pos.shape[0],))
    grid_pos = np.concatenate(
        (grid_pos, np.expand_dims(z_grid, axis=1)), axis=1
    )

    # Normalize mesh_static_features to a list of tensors for zero_index
    # functions: hierarchical -> BufferList, non-hierarchical -> single tensor
    msf_as_list = (
        list(mesh_static_features) if hierarchical else [mesh_static_features]
    )

    # The plotting requires the edges to be non-zero-indexed
    # with grid indices following mesh indices
    m2g_edge_index = utils.zero_index_m2g(
        m2g_edge_index, msf_as_list, mesh_first=True, restore=True
    )

    g2m_edge_index = utils.zero_index_g2m(
        g2m_edge_index, msf_as_list, mesh_first=True, restore=True
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

        # Compute cumulative node offsets per level (in the concatenated
        # mesh_pos array, level-l nodes start at level_offsets[l])
        # This is needed as the zero-indexing is applied to each level in
        # in load_graph()
        level_sizes = [msf.shape[0] for msf in mesh_static_features]
        level_offsets = np.cumsum([0] + level_sizes[:-1])

        # Add intra-level mesh edges (m2m per level)
        # Edge indices are zero-indexed per level, so shift by level offset
        for level, level_ei in enumerate(m2m_edge_index):
            ei_shifted = level_ei.numpy() + level_offsets[level]
            edge_plot_list.append((ei_shifted, "blue", 1, f"M2M Level {level}"))

        # Add inter-level mesh edges (up/down connect adjacent levels)
        for level, level_up_ei in enumerate(mesh_up_edge_index):
            ei_up = level_up_ei.numpy().copy()
            # Row 0: source in level l, Row 1: target in level l+1
            ei_up[0] += level_offsets[level]
            ei_up[1] += level_offsets[level + 1]
            edge_plot_list.append(
                (ei_up, "green", 1, f"Mesh up {level}->{level + 1}")
            )

        for level, level_down_ei in enumerate(mesh_down_edge_index):
            ei_down = level_down_ei.numpy().copy()
            # Row 0: source in level l+1, Row 1: target in level l
            ei_down[0] += level_offsets[level + 1]
            ei_down[1] += level_offsets[level]
            edge_plot_list.append(
                (ei_down, "green", 1, f"Mesh down {level + 1}->{level}")
            )

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
        edge_start = node_pos[ei[0]]  # (M, 3)
        edge_end = node_pos[ei[1]]  # (M, 3)
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

    if not show_axis:
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    if save:
        fig.write_html(save, include_plotlyjs="cdn")

    return fig


def main():
    """Plot graph structure in 3D using plotly."""
    parser = ArgumentParser(
        description="Plot graph",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datastore_config_path",
        type=str,
        default="tests/datastore_examples/mdp/config.yaml",
        help="Path for the datastore config",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to plot",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Name of .html file to save interactive plot to",
    )
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="If the axis should be displayed",
    )

    args = parser.parse_args()
    _, datastore = load_config_and_datastore(
        config_path=args.datastore_config_path
    )

    xy = datastore.get_xy("state", stacked=True)  # (N_grid, 2)
    pos_max = np.max(np.abs(xy))
    grid_pos = xy / pos_max  # Divide by maximum coordinate

    # Load graph data
    graph_dir_path = os.path.join(datastore.root_path, "graph", args.graph)
    hierarchical, graph_ldict = utils.load_graph(graph_dir_path=graph_dir_path)

    fig = plot_graph(
        grid_pos=grid_pos,
        hierarchical=hierarchical,
        graph_ldict=graph_ldict,
        show_axis=args.show_axis,
        save=args.save,
    )

    if not args.save:
        fig.show()


if __name__ == "__main__":
    main()
