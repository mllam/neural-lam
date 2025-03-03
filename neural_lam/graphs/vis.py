# Third-party
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg


def plot_graph(edge_index, from_node_pos, to_node_pos=None, title=None):
    """Plot graph using Plate Carree projection.

    Parameters
    ----------
    edge_index : tensor
        (2, N_edges) tensor of node indices.
    from_node_pos : tensor
        (N_nodes, 2) tensor containing longitudes and latitudes.
    to_node_pos : tensor, optional
        (N_nodes, 2) tensor containing longitudes and latitudes,
        or None (assumed same as from_node_pos).
    title : str, optional
        Title for the plot.

    Returns
    -------
    tuple
        (figure, axis) Matplotlib figure and axis objects.
    """
    if to_node_pos is None:
        # If to_node_pos is None it is same as from_node_pos
        to_node_pos = from_node_pos

    plate_carree = ccrs.PlateCarree()
    fig, axis = plt.subplots(subplot_kw={"projection": plate_carree})

    axis.coastlines(resolution="110m", color="gray", alpha=0.5)

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Move tensors to cpu and make numpy
    from_node_pos = from_node_pos.cpu().numpy()
    to_node_pos = to_node_pos.cpu().numpy()

    # Compute (in)-degrees
    from_degrees = (
        pyg.utils.degree(edge_index[0], num_nodes=from_node_pos.shape[0])
        .cpu()
        .numpy()
    )
    to_degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=to_node_pos.shape[0])
        .cpu()
        .numpy()
    )
    min_degree = min(from_degrees.min(), to_degrees.min())
    max_degree = max(from_degrees.max(), to_degrees.max())

    # Move edge_index to cpu and make numpy
    edge_index = edge_index.cpu().numpy()

    # Make all positions be in [-180, 180] for PC CRS
    from_node_pos[:, 0] = ((from_node_pos[:, 0] + 180.0) % 360) - 180
    to_node_pos[:, 0] = ((to_node_pos[:, 0] + 180.0) % 360) - 180

    assert (from_node_pos[:, 0] <= 180).all()
    assert (from_node_pos[:, 0] >= -180).all()
    assert (to_node_pos[:, 0] <= 180).all()
    assert (to_node_pos[:, 0] >= -180).all()

    # Plot edges
    from_pos = from_node_pos[edge_index[0]]  # (M/2, 2)
    to_pos = to_node_pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1, transform=plate_carree
        )
    )

    # Plot (sender) nodes
    node_scatter = axis.scatter(
        from_node_pos[:, 0],
        from_node_pos[:, 1],
        c=from_degrees,
        s=8,
        marker=".",
        zorder=2,
        cmap="viridis",
        clim=None,
        vmin=min_degree,
        vmax=max_degree,
        transform=plate_carree,
    )

    # Plot (receiver) nodes
    node_scatter = axis.scatter(
        to_node_pos[:, 0],
        to_node_pos[:, 1],
        c=to_degrees,
        s=20,
        marker="P",
        zorder=3,
        cmap="viridis",
        clim=None,
        vmin=min_degree,
        vmax=max_degree,
        transform=plate_carree,
    )

    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")

    axis.set_global()

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis
