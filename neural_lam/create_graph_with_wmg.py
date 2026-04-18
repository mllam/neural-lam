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


def _estimate_grid_node_spacing(xy):
    """Estimate the average grid node spacing from grid coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Grid coordinates of shape ``(N, 2)``.

    Returns
    -------
    float
        Estimated average grid node spacing in coordinate units.
    """
    x_range = np.ptp(xy[:, 0])
    y_range = np.ptp(xy[:, 1])
    n_points = len(xy)
    # avg grid spacing ≈ sqrt(area / n_points)
    return float(np.sqrt(x_range * y_range / n_points))


def create_graph_from_datastore(
    datastore,
    output_root_path,
    archetype="keisler",
    mesh_node_distance=None,
    grid_mesh_spacing_ratio=3.0,
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
        automatically estimated as ``grid_mesh_spacing_ratio * grid_spacing``.
    grid_mesh_spacing_ratio : float
        Ratio of mesh node distance to grid node spacing. Only used when
        ``mesh_node_distance`` is None. Default is 3.0.
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

    xy = datastore.get_xy(category="state", stacked=True)
    xy = np.array(xy)

    if mesh_node_distance is None:
        grid_spacing = _estimate_grid_node_spacing(xy)
        mesh_node_distance = grid_spacing * grid_mesh_spacing_ratio

    # Build keyword arguments for the archetype function.
    # return_components=True is required because
    # wmg.save.to_torch_tensors_on_disk() expects the graph as
    # separate g2m, m2g and m2m sub-graph components
    # rather than a single merged graph.
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

    wmg.save.to_torch_tensors_on_disk(
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
        "If not set, estimated automatically from grid spacing "
        "and --grid_mesh_spacing_ratio.",
    )
    parser.add_argument(
        "--grid_mesh_spacing_ratio",
        type=float,
        default=3.0,
        help="Ratio of mesh node distance to grid node spacing. "
        "Only used when --mesh_node_distance is not set.",
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
        grid_mesh_spacing_ratio=args.grid_mesh_spacing_ratio,
        level_refinement_factor=args.level_refinement_factor,
        max_num_levels=args.max_num_levels,
    )


if __name__ == "__main__":
    cli()
