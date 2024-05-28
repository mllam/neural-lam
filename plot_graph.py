# Standard library
from argparse import ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch_geometric as pyg
import trimesh
from tqdm import tqdm
from trimesh.primitives import Box

# First-party
from neural_lam import config, utils

MESH_HEIGHT = 0.1
MESH_LEVEL_DIST = 0.05
GRID_HEIGHT = 0


def create_cubes_for_nodes(nodes, size=0.002):
    """Create cubes for each node in the graph."""
    cube_meshes = []
    for node in tqdm(nodes, desc="Creating cubes"):
        cube = Box(extents=[size, size, size])
        cube.apply_translation(node)
        cube_meshes.append(cube)
    return cube_meshes


def export_to_3d_model(node_pos, edge_plot_list, filename):
    """Export the graph to a 3D model."""
    paths = []
    filtered_edge_plot_list = [
        item for item in edge_plot_list if item[3] not in ["M2G", "G2M"]
    ]

    unique_node_indices = set()
    for ei, _, _, _ in filtered_edge_plot_list:
        unique_node_indices.update(ei.flatten())

    filtered_node_positions = node_pos[np.array(list(unique_node_indices))]

    for ei, _, _, _ in filtered_edge_plot_list:
        edge_start = filtered_node_positions[ei[0]]
        edge_end = filtered_node_positions[ei[1]]
        for start, end in zip(edge_start, edge_end):
            if not (np.isnan(start).any() or np.isnan(end).any()):
                paths.append((start, end))

    meshes = []
    for start, end in tqdm(paths, desc="Creating meshes"):
        offset_xyz = np.array([2e-4, 2e-4, 2e-4])
        dummy_vertex = end + offset_xyz
        vertices = [start, end, dummy_vertex]
        faces = [[0, 1, 2]]
        color_vertices = [[255, 179, 71], [255, 179, 71], [255, 179, 71]]
        # color_faces = [[0, 0, 0]]

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            # face_colors=color_faces,
            vertex_colors=color_vertices,
        )
        meshes.append(mesh)

    node_spheres = create_cubes_for_nodes(filtered_node_positions)

    scene = trimesh.Scene()
    for mesh in meshes:
        scene.add_geometry(mesh)
    for sphere in node_spheres:
        scene.add_geometry(sphere)

    scene.export(filename, file_type="ply")


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
    parser.add_argument(
        "--export",
        type=str,
        help="Name of .obj file to export 3D model to (default: None)",
    )

    args = parser.parse_args()

    hierarchical, graph_ldict = utils.load_graph(args.graph)
    g2m_edge_index, m2g_edge_index, m2m_edge_index = (
        graph_ldict["g2m_edge_index"],
        graph_ldict["m2g_edge_index"],
        graph_ldict["m2m_edge_index"],
    )
    mesh_up_edge_index, mesh_down_edge_index = (
        graph_ldict["mesh_up_edge_index"],
        graph_ldict["mesh_down_edge_index"],
    )
    mesh_static_features = graph_ldict["mesh_static_features"]

    config_loader = config.Config(args.data_config)
    xy = config_loader.get_nwp_xy()
    grid_xy = xy.transpose(1, 2, 0).reshape(-1, 2)
    pos_max = np.max(np.abs(grid_xy))
    grid_pos = grid_xy / pos_max

    z_grid = GRID_HEIGHT * np.ones((grid_pos.shape[0],))
    grid_pos = np.concatenate(
        (grid_pos, np.expand_dims(z_grid, axis=1)), axis=1
    )

    edge_plot_list = [
        (m2g_edge_index.numpy(), "black", 0.4, "M2G"),
        (g2m_edge_index.numpy(), "black", 0.4, "G2M"),
    ]

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

        edge_plot_list += [
            (level_ei.numpy(), "blue", 1, f"M2M Level {level}")
            for level, level_ei in enumerate(m2m_edge_index)
        ]

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

    node_pos = np.concatenate((mesh_pos, grid_pos), axis=0)

    data_objs = []
    for ei, col, width, label in edge_plot_list:
        edge_start = node_pos[ei[0]]
        edge_end = node_pos[ei[1]]
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
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")

    if args.export:
        export_to_3d_model(node_pos, edge_plot_list, args.export)


if __name__ == "__main__":
    main()
