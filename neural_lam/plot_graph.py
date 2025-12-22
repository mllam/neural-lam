# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import graph_utils as gutils

GRID_RADIUS = 1
# https://community.plotly.com/t/whats-the-efficient-way-to-create-3d-scatter-plot-for-millions-of-points/60965/4
NODE_PLOT_LIMIT = 100000  # Limit on number of points to plot before subsampling


def make_earth(radius, resolution_reduction=1.0):
    """
    Plotly earth from
    https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166

    radius: radius of earth in plot
    resolution: float, percentage of full resolution
    """
    earth_colorscale = [
        [0.0, "rgb(30, 59, 117)"],
        [0.1, "rgb(46, 68, 21)"],
        [0.2, "rgb(74, 96, 28)"],
        [0.3, "rgb(115,141,90)"],
        [0.4, "rgb(122, 126, 75)"],
        [0.6, "rgb(122, 126, 75)"],
        [0.7, "rgb(141,115,96)"],
        [0.8, "rgb(223, 197, 170)"],
        [0.9, "rgb(237,214,183)"],
        [1.0, "rgb(255, 255, 255)"],
    ]
    texture_path = "figures/earth_texture.jpeg"
    img = Image.open(texture_path)

    # Calculate new width to maintain aspect ratio
    new_width = int(img.width * resolution_reduction)
    new_height = int(img.height * resolution_reduction)

    # Resize image preserving aspect ratio
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    texture = np.asarray(img_resized).T

    N_lon = int(texture.shape[0])
    N_lat = int(texture.shape[1])
    theta = np.linspace(-np.pi, np.pi, N_lon)
    phi = np.linspace(0, np.pi, N_lat)

    # Set up coordinates for points on the sphere
    x0 = radius * np.outer(np.cos(theta), np.sin(phi))
    y0 = radius * np.outer(np.sin(theta), np.sin(phi))
    z0 = radius * np.outer(np.ones(N_lon), np.cos(phi))

    return go.Surface(
        x=x0,
        y=y0,
        z=z0,
        surfacecolor=texture,
        colorscale=earth_colorscale,
        name="Earth",
        showscale=False,
        showlegend=True,
    )


def create_edge_plot(
    edge_index,
    from_node_lat_lon,
    to_node_lat_lon,
    label,
    color="blue",
    width=1,
    from_radius=1,
    to_radius=1,
    pos_filter_func=None,
):
    """
    Create a plotly object showing edges

    edge_index: (2, M)
    from_node_lat_lon: (N, 2), positions of sender nodes
    to_node_lat_lon: (N, 2), positions of receiver nodes
    label: str, label of plot object
    """
    from_node_cart = (
        gutils.node_lat_lon_to_cart(from_node_lat_lon) * from_radius
    )
    to_node_cart = gutils.node_lat_lon_to_cart(to_node_lat_lon) * to_radius

    edge_start = from_node_cart[edge_index[0]]  # (M, 2)
    edge_end = to_node_cart[edge_index[1]]  # (M, 2)

    if pos_filter_func is not None:
        # Filter edges
        edge_start_lat_lon = from_node_lat_lon[edge_index[0]]  # (M, 2)
        edge_end_lat_lon = to_node_lat_lon[edge_index[1]]  # (M, 2)

        edge_mask = np.logical_and(
            pos_filter_func(edge_start_lat_lon),
            pos_filter_func(edge_end_lat_lon),
        )
        edge_start = edge_start[edge_mask]
        edge_end = edge_end[edge_mask]

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


def create_node_plot(
    node_lat_lon, label, color="blue", size=1, radius=1, pos_filter_func=None
):
    """
    Create a plotly object showing nodes

    node_lat_lon: (N, 2)
    label: str, label of plot object
    """
    node_pos = gutils.node_lat_lon_to_cart(node_lat_lon) * radius
    if pos_filter_func is not None:
        # Filter nodes before plotting
        node_pos = node_pos[pos_filter_func(node_lat_lon)]

    # Plotly 3d can not render large amounts of points in some browsers, so
    # for very large node sets we need to somehow subsample it before plotting.
    # This is a simple solution
    num_nodes = node_pos.shape[0]
    subsample = num_nodes > NODE_PLOT_LIMIT
    if subsample:
        # Figure out how much to subsample by
        subsampling_factor = int(num_nodes / NODE_PLOT_LIMIT)
        node_pos = node_pos[::subsampling_factor]  # Simple subsampling

    return go.Scatter3d(
        x=node_pos[:, 0],
        y=node_pos[:, 1],
        z=node_pos[:, 2],
        mode="markers",
        marker={"color": color, "size": size},
        name=f"{label} (subsampled)" if subsample else label,
    )


def main():
    """Plot graph structure in 3D using plotly."""
    parser = ArgumentParser(
        description="Plot graph",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Name of saved graph to plot",
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
    parser.add_argument(
        "--corner_filter_radius",
        type=float,
        help="Filter plotted objects to within given radius of interior corner",
    )
    # Geometry
    parser.add_argument(
        "--mesh_height",
        type=float,
        default=0.1,
        help="Height of mesh over grid",
    )
    parser.add_argument(
        "--mesh_level_dist",
        type=float,
        default=0.3,
        help="Distance between mesh levels",
    )
    parser.add_argument(
        "--edge_width",
        type=float,
        default=0.4,
        help="Width of edges",
    )
    parser.add_argument(
        "--mesh_edge_width",
        type=float,
        help="Width of mesh edges, if different than --edge_width",
    )
    parser.add_argument(
        "--grid_node_size",
        type=float,
        default=1.0,
        help="Size of grid nodes",
    )
    parser.add_argument(
        "--mesh_node_size",
        type=float,
        default=2.5,
        help="Size of mesh nodes",
    )
    # Colors
    parser.add_argument(
        "--g2m_color",
        type=str,
        default="black",
        help="Color of g2m edges",
    )
    parser.add_argument(
        "--m2g_color",
        type=str,
        default="black",
        help="Color of m2g edges",
    )
    parser.add_argument(
        "--grid_color",
        type=str,
        default="black",
        help="Color of grid nodes (interior in case of LAM setup)",
    )
    parser.add_argument(
        "--boundary_grid_color",
        type=str,
        default="black",
        help="Color of boundary grid nodes in LAM setup",
    )
    parser.add_argument(
        "--mesh_color",
        type=str,
        default="blue",
        help="Color of mesh nodes and edges",
    )
    # Earth
    parser.add_argument(
        "--texture_resolution",
        type=float,
        default=0.5,
        help="Resolution of texture on earth, 1.0 is full resolution "
        "(high resolution can be slow)",
    )

    args = parser.parse_args()

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    _, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )
    boundary_forced = datastore_boundary is not None

    # Load graph data
    graph_dir_path = os.path.join(
        datastore.root_path, "graphs", args.graph_name
    )
    hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=graph_dir_path,
        datastore=datastore,
    )
    # Turn all to numpy
    (g2m_edge_index, m2g_edge_index) = (
        graph_ldict["g2m_edge_index"].numpy(),
        graph_ldict["m2g_edge_index"].numpy(),
    )

    # Plotting is in 3d, with lat-lons
    grid_lat_lon = utils.get_stacked_lat_lons(datastore, datastore_boundary)
    # (num_nodes_full, 3)

    # Optionally create corner filter
    if args.corner_filter_radius is not None:
        # Prep for filtering
        interior_lat_lon = datastore.get_lat_lon(category="state")
        # Define corner in terms of last point
        # Note: Could we do something more clever?
        corner = interior_lat_lon[-1]
        lon_corner, lat_corner = corner

        def corner_filter_func(pos_lat_lon):
            """
            pos is (N, 2)
            measure distance using haversine dist
            """

            lon_pos = pos_lat_lon[:, 0]
            lat_pos = pos_lat_lon[:, 1]

            lon_rad_corner, lat_rad_corner, lon_rad_pos, lat_rad_pos = map(
                np.radians, [lon_corner, lat_corner, lon_pos, lat_pos]
            )

            dlon_rad = lon_rad_pos - lon_rad_corner
            dlat_rad = lat_rad_pos - lat_rad_corner

            hav_interm = (
                np.sin(dlat_rad / 2.0) ** 2
                + np.cos(lat_rad_corner)
                * np.cos(lat_rad_pos)
                * np.sin(dlon_rad / 2.0) ** 2
            )
            hav_rad = 2 * np.arcsin(np.sqrt(hav_interm))
            hav_m = 6378137.0 * hav_rad

            return hav_m <= args.corner_filter_radius

    else:
        corner_filter_func = None

    mesh_edge_width = (
        args.edge_width
        if args.mesh_edge_width is None
        else args.mesh_edge_width
    )

    # Add plotting objects to this list
    data_objs = []

    # Plot grid nodes
    if boundary_forced:
        # Create separate plot objects for interior and boundary
        data_objs.append(
            create_node_plot(
                datastore.get_lat_lon(category="state"),
                "Interior grid Nodes",
                color=args.grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
        data_objs.append(
            create_node_plot(
                datastore_boundary.get_lat_lon(category="forcing"),
                "Boundary grid Nodes",
                color=args.boundary_grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
    else:
        # All grid nodes together
        data_objs.append(
            create_node_plot(
                grid_lat_lon,
                "Grid Nodes",
                color=args.grid_color,
                radius=GRID_RADIUS,
                size=args.grid_node_size,
                pos_filter_func=corner_filter_func,
            )
        )

    # Radius
    mesh_radius = GRID_RADIUS + args.mesh_height

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        # Make edge_index to numpy
        def tensor_list_to_numpy(tensor_list):
            """Helper function to make list of tensors numpy arrays"""
            return [elem.numpy() for elem in tensor_list]

        m2m_edge_index = tensor_list_to_numpy(graph_ldict["m2m_edge_index"])
        mesh_lat_lon_level = tensor_list_to_numpy(graph_ldict["mesh_lat_lon"])
        mesh_up_edge_index = tensor_list_to_numpy(
            graph_ldict["mesh_up_edge_index"]
        )
        mesh_down_edge_index = tensor_list_to_numpy(
            graph_ldict["mesh_down_edge_index"]
        )

        # Iterate over levels, adding all nodes and edges
        for bot_level_i, intra_ei in enumerate(
            m2m_edge_index,
        ):
            # Extract position and radius
            top_level_i = bot_level_i + 1
            bot_pos = mesh_lat_lon_level[bot_level_i]
            bot_radius = mesh_radius + bot_level_i * args.mesh_level_dist

            # Mesh nodes at bottom level
            data_objs.append(
                create_node_plot(
                    bot_pos,
                    f"Mesh level {bot_level_i} nodes",
                    color=args.mesh_color,
                    radius=bot_radius,
                    size=args.mesh_node_size,
                    pos_filter_func=corner_filter_func,
                )
            )
            # Intra-level edges at bottom level
            data_objs.append(
                create_edge_plot(
                    intra_ei,
                    bot_pos,
                    bot_pos,
                    f"Mesh level {bot_level_i} edges",
                    color=args.mesh_color,
                    width=mesh_edge_width,
                    from_radius=bot_radius,
                    to_radius=bot_radius,
                    pos_filter_func=corner_filter_func,
                )
            )

            # Do add include up/down edges for top level
            if top_level_i < len(m2m_edge_index):
                up_ei = mesh_up_edge_index[bot_level_i]
                down_ei = mesh_down_edge_index[bot_level_i]
                top_pos = mesh_lat_lon_level[top_level_i]
                top_radius = mesh_radius + (top_level_i) * args.mesh_level_dist

                # Up edges
                data_objs.append(
                    create_edge_plot(
                        up_ei,
                        bot_pos,
                        top_pos,
                        f"Mesh up {bot_level_i}->{top_level_i} edges",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=bot_radius,
                        to_radius=top_radius,
                        pos_filter_func=corner_filter_func,
                    )
                )
                # Down edges
                data_objs.append(
                    create_edge_plot(
                        down_ei,
                        top_pos,
                        bot_pos,
                        f"Mesh down {top_level_i}->{bot_level_i} edges",
                        color=args.mesh_color,
                        width=mesh_edge_width,
                        from_radius=top_radius,
                        to_radius=bot_radius,
                        pos_filter_func=corner_filter_func,
                    )
                )

        # Connect g2m and m2g only to bottom level
        grid_con_lat_lon = mesh_lat_lon_level[0]
    else:
        mesh_lat_lon = graph_ldict["mesh_lat_lon"].numpy()

        # Non-hierarchical
        m2m_edge_index = graph_ldict["m2m_edge_index"].numpy()
        # TODO Degree-dependent node size option?
        #  mesh_degrees = pyg.utils.degree(m2m_edge_index[1]).numpy()
        #  mesh_node_size = mesh_degrees / 2

        data_objs.append(
            create_node_plot(
                mesh_lat_lon,
                "Mesh Nodes",
                radius=mesh_radius,
                color=args.mesh_color,
                size=args.mesh_node_size,
                pos_filter_func=corner_filter_func,
            )
        )
        data_objs.append(
            create_edge_plot(
                m2m_edge_index,
                mesh_lat_lon,
                mesh_lat_lon,
                "Mesh Edges",
                from_radius=mesh_radius,
                to_radius=mesh_radius,
                color=args.mesh_color,
                width=mesh_edge_width,
                pos_filter_func=corner_filter_func,
            )
        )

        grid_con_lat_lon = mesh_lat_lon

    # Plot G2M
    data_objs.append(
        create_edge_plot(
            g2m_edge_index,
            grid_lat_lon,
            grid_con_lat_lon,
            "G2M Edges",
            color=args.g2m_color,
            width=args.edge_width,
            from_radius=GRID_RADIUS,
            to_radius=mesh_radius,
            pos_filter_func=corner_filter_func,
        )
    )

    # Plot M2G
    data_objs.append(
        create_edge_plot(
            m2g_edge_index,
            grid_con_lat_lon,
            grid_lat_lon,
            "M2G Edges",
            color=args.m2g_color,
            width=args.edge_width,
            from_radius=mesh_radius,
            to_radius=GRID_RADIUS,
            pos_filter_func=corner_filter_func,
        )
    )

    # Plot earth
    data_objs.append(
        make_earth(radius=1, resolution_reduction=args.texture_resolution)
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
