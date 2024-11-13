# Standard library
import argparse
import os

# Third-party
import numpy as np
import weather_model_graphs as wmg

# Local
from . import config, utils

WMG_ARCHETYPES = {
    "keisler": wmg.create.archetype.create_keisler_graph,
    "graphcast": wmg.create.archetype.create_graphcast_graph,
    "hierarchical": wmg.create.archetype.create_oskarsson_hierarchical_graph,
}


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description="Graph generation using WMG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs and outputs
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file",
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

    # Load grid positions
    config_loader = config.Config.from_file(args.data_config)

    # TODO Do not get normalised positions
    coords = utils.get_reordered_grid_pos(config_loader.dataset.name).numpy()
    # (num_nodes_full, 2)

    # Construct mask
    static_data = utils.load_static_data(config_loader.dataset.name)
    decode_mask = np.concatenate(
        (
            np.ones(static_data["grid_static_features"].shape[0], dtype=bool),
            np.zeros(
                static_data["boundary_static_features"].shape[0], dtype=bool
            ),
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
        "projection": None,
        "decode_mask": decode_mask,
    }
    if args.archetype != "keisler":
        # Add additional multi-level kwargs
        create_kwargs.update(
            {
                "level_refinement_factor": args.level_refinement_factor,
                "max_num_levels": args.max_num_levels,
            }
        )

    graph = archetype_create_func(**create_kwargs)
    graph_comp = wmg.split_graph_by_edge_attribute(graph, attr="component")

    print("Created graph:")
    for name, subgraph in graph_comp.items():
        print(f"{name}: {subgraph}")

    # Save graph
    os.makedirs(args.output_dir, exist_ok=True)
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
                            output_directory=args.output_dir,
                        )
                    else:
                        # up and down directions
                        wmg.save.to_pyg(
                            graph=graph,
                            name=f"mesh_{direction}",
                            list_from_attribute="levels",
                            edge_features=["len", "vdiff"],
                            output_directory=args.output_dir,
                        )
            else:
                wmg.save.to_pyg(
                    graph=graph,
                    name=component,
                    list_from_attribute="dummy",  # Note: Needed to output list
                    edge_features=["len", "vdiff"],
                    output_directory=args.output_dir,
                )
        else:
            wmg.save.to_pyg(
                graph=graph,
                name=component,
                edge_features=["len", "vdiff"],
                output_directory=args.output_dir,
            )


if __name__ == "__main__":
    main()
