# Standard library
import argparse
import os

# Third-party
import cartopy.crs as ccrs
import weather_model_graphs as wmg

# Local
from . import utils
from .config import load_config_and_datastores

WMG_ARCHETYPES = {
    "keisler": wmg.create.archetype.create_keisler_graph,
    "graphcast": wmg.create.archetype.create_graphcast_graph,
    "hierarchical": wmg.create.archetype.create_oskarsson_hierarchical_graph,
}


def main(input_args=None):
    """
    Build rectangular graph from archetype, using cmd-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rectangular graph generation using weather-models-graph",
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
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graphs",
        help="Directory to save graph to",
    )

    # Graph structure
    parser.add_argument(
        "--archetype",
        type=str,
        default="keisler",
        help="Archetype to use to create graph "
        "(keisler/graphcast/hierarchical)",
    )
    parser.add_argument(
        "--mesh_node_distance",
        type=float,
        default=3.0,
        help="Distance between created mesh nodes",
    )
    parser.add_argument(
        "--level_refinement_factor",
        type=int,
        default=3,
        help="Refinement factor between grid points and bottom level of "
        "mesh hierarchy",
    )
    parser.add_argument(
        "--max_num_levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up",
    )
    args = parser.parse_args(input_args)

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"
    assert (
        args.graph_name is not None
    ), "Specify the name to save graph as with --graph_name"

    _, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    create_kwargs = {
        "mesh_node_distance": args.mesh_node_distance,
    }

    if args.archetype != "keisler":
        # Add additional multi-level kwargs
        create_kwargs.update(
            {
                "level_refinement_factor": args.level_refinement_factor,
                "max_num_levels": args.max_num_levels,
            }
        )

    return build_graph_from_archetype(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_name=args.graph_name,
        archetype=args.archetype,
        **create_kwargs,
    )


def _build_wmg_graph(
    datastore,
    datastore_boundary,
    graph_build_func,
    kwargs,
    graph_name,
    dir_save_path=None,
):
    """
    Build a graph using WMG in a way that's compatible with neural-lam.
    Given datastores are used for coordinates and decode masking.
    The given graph building function from WMG should be used, with kwargs.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary forcing
    graph_build_func
        Function from WMG to use to build graph
    kwargs : dict
        Keyword arguments to feed to graph_build_func. Should not include
        coords, coords_crs, graph_crs, return_components or decode_mask, as
        these are here derived in a consistent way from the datastores.
    graph_name : str
        Name to save the graph as.
    dir_save_path : str or None
        Path to directory where graph should be saved, in directory graph_name.
        If None, save in "graphs" directory in the root directory of datastore.
    """

    for derived_kwarg in (
        "coords",
        "coords_crs",
        "graph_crs",
        "return_components",
        "decode_mask",
    ):
        assert derived_kwarg not in kwargs, (
            f"Argument {derived_kwarg} should not be manually given when "
            "building rectangular graph."
        )

    # Load grid positions
    coords = utils.get_stacked_lat_lons(datastore, datastore_boundary)
    # (num_nodes_full, 2)
    # Project using crs from datastore for graph building
    coords_crs = ccrs.PlateCarree()
    graph_crs = datastore.coords_projection

    if datastore_boundary is None:
        # No mask
        decode_mask = None
    else:
        # Construct mask to decode only to interior
        decode_mask = utils.get_interior_mask(datastore, datastore_boundary)

    # Set up all kwargs
    create_kwargs = {
        "coords": coords,
        "decode_mask": decode_mask,
        "graph_crs": graph_crs,
        "coords_crs": coords_crs,
        "return_components": True,
    }
    create_kwargs.update(kwargs)

    # Build graph
    graph_comp = graph_build_func(**create_kwargs)

    print("Created graph:")
    for name, subgraph in graph_comp.items():
        print(f"{name}: {subgraph}")

    # Need to know if hierarchical for saving
    hierarchical = (graph_build_func == WMG_ARCHETYPES["hierarchical"]) or (
        "m2m_connectivity" in kwargs
        and kwargs["m2m_connectivity"] == "hierarchical"
    )

    # Save graph
    if dir_save_path is None:
        graph_dir_path = os.path.join(datastore.root_path, "graphs", graph_name)
    else:
        graph_dir_path = os.path.join(dir_save_path, graph_name)

    os.makedirs(graph_dir_path, exist_ok=True)
    for component, graph in graph_comp.items():
        # This seems like a bit of a hack, maybe better if saving in wmg
        # was made consistent with nl
        if component == "m2m":
            if hierarchical:
                # Split by direction
                m2m_direction_comp = wmg.split_graph_by_edge_attribute(
                    graph, attr="direction"
                )
                for direction, dir_graph in m2m_direction_comp.items():
                    if direction == "same":
                        # Name just m2m to be consistent with non-hierarchical
                        wmg.save.to_pyg(
                            graph=dir_graph,
                            name="m2m",
                            list_from_attribute="level",
                            edge_features=["len", "vdiff"],
                            output_directory=graph_dir_path,
                        )
                    else:
                        # up and down directions
                        wmg.save.to_pyg(
                            graph=dir_graph,
                            name=f"mesh_{direction}",
                            list_from_attribute="levels",
                            edge_features=["len", "vdiff"],
                            output_directory=graph_dir_path,
                        )
            else:
                wmg.save.to_pyg(
                    graph=graph,
                    name=component,
                    list_from_attribute="dummy",  # Note: Needed to output list
                    edge_features=["len", "vdiff"],
                    output_directory=graph_dir_path,
                )
        else:
            wmg.save.to_pyg(
                graph=graph,
                name=component,
                edge_features=["len", "vdiff"],
                output_directory=graph_dir_path,
            )


def build_graph_from_archetype(
    datastore,
    datastore_boundary,
    graph_name,
    archetype,
    dir_save_path=None,
    **kwargs,
):
    """
    Function that builds graph using wmg archetype.
    Uses archetype functions from wmg.create.archetype with kwargs being passed
    on directly to those functions.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary forcing
    graph_name : str
        Name to save the graph as.
    archetype : str
        Archetype to build. Must be one of "keisler", "graphcast"
        or "hierarchical"
    dir_save_path : str or None
        Path to directory where graph should be saved, in directory graph_name.
        If None, save in "graphs" directory in the root directory of datastore.
    **kwargs
        Keyword arguments that are passed on to
        wmg.create.base.create_all_graph_components. See WMG for accepted
        values for these.
    """

    assert archetype in WMG_ARCHETYPES, f"Unknown archetype: {archetype}"
    archetype_create_func = WMG_ARCHETYPES[archetype]

    return _build_wmg_graph(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_build_func=archetype_create_func,
        graph_name=graph_name,
        dir_save_path=dir_save_path,
        kwargs=kwargs,
    )


def build_graph(
    datastore, datastore_boundary, graph_name, dir_save_path=None, **kwargs
):
    """
    Function that can be used for more fine-grained control of graph
    construction. Directly uses wmg.create.base.create_all_graph_components,
    with kwargs being passed on directly to there.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore representing interior region of grid
    datastore_boundary : BaseDatastore or None
        Datastore representing boundary region, or None if no boundary forcing
    graph_name : str
        Name to save the graph as.
    dir_save_path : str or None
        Path to directory where graph should be saved, in directory graph_name.
        If None, save in "graphs" directory in the root directory of datastore.
    **kwargs
        Keyword arguments that are passed on to
        wmg.create.base.create_all_graph_components. See WMG for accepted
        values for these.
    """
    return _build_wmg_graph(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        graph_build_func=wmg.create.base.create_all_graph_components,
        graph_name=graph_name,
        dir_save_path=dir_save_path,
        kwargs=kwargs,
    )


if __name__ == "__main__":
    main()
