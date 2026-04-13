# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import numpy as np
import weather_model_graphs as wmg

# Local
from .config import load_config_and_datastore
from .datastore.base import BaseRegularGridDatastore

ARCHETYPE_FUNCTIONS = {
    "keisler": wmg.create.archetype.create_keisler_graph,
    "graphcast": wmg.create.archetype.create_graphcast_graph,
    "hierarchical": wmg.create.archetype.create_oskarsson_hierarchical_graph,
}


def _estimate_mesh_node_distance(xy):
    """Estimate a reasonable mesh node distance from grid coordinates.

    Uses the average grid spacing to produce a mesh that is roughly 3x
    coarser than the grid, similar to the default behaviour of the old
    ``create_graph.py`` script.

    Parameters
    ----------
    xy : np.ndarray
        Grid coordinates of shape ``(N, 2)``.

    Returns
    -------
    float
        Estimated mesh node distance in coordinate units.
    """
    x_range = np.ptp(xy[:, 0])
    y_range = np.ptp(xy[:, 1])
    n_points = len(xy)
    # avg grid spacing ≈ sqrt(area / n_points)
    avg_spacing = np.sqrt(x_range * y_range / n_points)
    # mesh is ~3x coarser than the grid
    return float(avg_spacing * 3)


def create_graph_from_datastore(
    datastore,
    output_root_path,
    archetype="keisler",
    mesh_node_distance=None,
    level_refinement_factor=3,
    max_num_levels=None,
):
    """Create graph using weather-model-graphs and save in neural-lam format.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore providing grid coordinates.
    output_root_path : str
        Directory where the .pt graph files will be saved.
    archetype : str
        Graph archetype to create: ``"keisler"``, ``"graphcast"``, or
        ``"hierarchical"``.
    mesh_node_distance : float or None
        Distance between created mesh nodes (in coordinate units). If None,
        automatically estimated from the grid spacing.
    level_refinement_factor : int
        Refinement factor between mesh hierarchy levels. Only used for
        ``"graphcast"`` and ``"hierarchical"`` archetypes.
    max_num_levels : int or None
        Maximum number of mesh hierarchy levels. Only used for ``"graphcast"``
        and ``"hierarchical"`` archetypes.
    """
    if not isinstance(datastore, BaseRegularGridDatastore):
        raise NotImplementedError(
            "Only graph creation for BaseRegularGridDatastore is supported"
        )

    if archetype not in ARCHETYPE_FUNCTIONS:
        raise ValueError(
            f"Unknown archetype '{archetype}'. "
            f"Must be one of: {list(ARCHETYPE_FUNCTIONS.keys())}"
        )

    xy = datastore.get_xy(category="state", stacked=False)

    # wmg expects coords as 2D array of shape (num_nodes, 2), but the
    # datastore may return a 3D array of shape (Nx, Ny, 2) when
    # stacked=False.  Reshape to (N, 2) for wmg.
    xy = np.array(xy)
    if xy.ndim == 3:
        xy = xy.reshape(-1, 2)

    if mesh_node_distance is None:
        mesh_node_distance = _estimate_mesh_node_distance(xy)

    # Build keyword arguments for the archetype function
    archetype_kwargs = dict(
        coords=xy,
        mesh_node_distance=mesh_node_distance,
        return_components=True,
    )

    # Only multiscale/hierarchical archetypes accept these parameters
    if archetype in ("graphcast", "hierarchical"):
        archetype_kwargs["level_refinement_factor"] = level_refinement_factor
        archetype_kwargs["max_num_levels"] = max_num_levels

    archetype_fn = ARCHETYPE_FUNCTIONS[archetype]
    graph_components = archetype_fn(**archetype_kwargs)

    hierarchical = archetype == "hierarchical"

    wmg.save.to_neural_lam(
        graph_components=graph_components,
        output_directory=output_root_path,
        hierarchical=hierarchical,
    )


def cli(input_args=None):
    """Command-line interface for graph creation using weather-model-graphs."""
    parser = ArgumentParser(
        description="Graph generation for neural-lam using "
        "weather-model-graphs (wmg)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to neural-lam configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multiscale",
        help="Name to save graph as (used as subdirectory name)",
    )
    parser.add_argument(
        "--archetype",
        type=str,
        default="keisler",
        choices=["keisler", "graphcast", "hierarchical"],
        help="Graph archetype to create",
    )
    parser.add_argument(
        "--mesh_node_distance",
        type=float,
        default=None,
        help="Distance between mesh nodes (in coordinate units). "
        "If not set, estimated automatically from grid spacing.",
    )
    parser.add_argument(
        "--level_refinement_factor",
        type=int,
        default=3,
        help="Refinement factor between mesh hierarchy levels "
        "(only used for graphcast and hierarchical)",
    )
    parser.add_argument(
        "--max_num_levels",
        type=int,
        default=None,
        help="Maximum number of mesh levels "
        "(only used for graphcast and hierarchical)",
    )
    args = parser.parse_args(input_args)

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    # Load neural-lam configuration and datastore to use
    _, datastore = load_config_and_datastore(config_path=args.config_path)

    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=os.path.join(datastore.root_path, "graph", args.name),
        archetype=args.archetype,
        mesh_node_distance=args.mesh_node_distance,
        level_refinement_factor=args.level_refinement_factor,
        max_num_levels=args.max_num_levels,
    )


if __name__ == "__main__":
    cli()
