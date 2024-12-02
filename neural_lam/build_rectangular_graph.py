# Standard library
import argparse
import os

# Third-party
import numpy as np
import weather_model_graphs as wmg

# Local
from . import utils
from .config import load_config_and_datastore

WMG_ARCHETYPES = {
    "keisler": wmg.create.archetype.create_keisler_graph,
    "graphcast": wmg.create.archetype.create_graphcast_graph,
    "hierarchical": wmg.create.archetype.create_oskarsson_hierarchical_graph,
}


def main(input_args=None):
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
        type=float,
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

    _, datastore = load_config_and_datastore(config_path=args.config_path)

    # Load grid positions
    # TODO Do not get normalised positions
    coords = utils.get_reordered_grid_pos(datastore).numpy()
    # (num_nodes_full, 2)

    # Construct mask
    num_full_grid = coords.shape[0]
    num_boundary = datastore.boundary_mask.to_numpy().sum()
    num_interior = num_full_grid - num_boundary
    decode_mask = np.concatenate(
        (
            np.ones(num_interior, dtype=bool),
            np.zeros(num_boundary, dtype=bool),
        ),
        axis=0,
    )

    # Build graph
    assert (
        args.archetype in WMG_ARCHETYPES
    ), f"Unknown archetype: {args.archetype}"
    archetype_create_func = WMG_ARCHETYPES[args.archetype]

    create_kwargs = {
        "coords": coords,
        "mesh_node_distance": args.mesh_node_distance,
        "decode_mask": decode_mask,
        "return_components": True,
    }
    if args.archetype != "keisler":
        # Add additional multi-level kwargs
        create_kwargs.update(
            {
                "level_refinement_factor": args.level_refinement_factor,
                "max_num_levels": args.max_num_levels,
            }
        )

    graph_comp = archetype_create_func(**create_kwargs)

    print("Created graph:")
    for name, subgraph in graph_comp.items():
        print(f"{name}: {subgraph}")

    # Save graph
    graph_dir_path = os.path.join(
        datastore.root_path, "graphs", args.graph_name
    )
    os.makedirs(graph_dir_path, exist_ok=True)
    for component, graph in graph_comp.items():
        # This seems like a bit of a hack, maybe better if saving in wmg
        # was made consistent with nl
        if component == "m2m":
            if args.archetype == "hierarchical":
                # Split by direction
                m2m_direction_comp = wmg.split_graph_by_edge_attribute(
                    graph, attr="direction"
                )
                for direction, graph in m2m_direction_comp.items():
                    if direction == "same":
                        # Name just m2m to be consistent with non-hierarchical
                        wmg.save.to_pyg(
                            graph=graph,
                            name="m2m",
                            list_from_attribute="level",
                            edge_features=["len", "vdiff"],
                            output_directory=graph_dir_path,
                        )
                    else:
                        # up and down directions
                        wmg.save.to_pyg(
                            graph=graph,
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


if __name__ == "__main__":
    main()
