# Standard library
from pathlib import Path

# Third-party
import pytest

# First-party
from neural_lam.build_rectangular_graph import (
    build_graph,
    build_graph_from_archetype,
)
from neural_lam.datastore import DATASTORES
from tests.conftest import (
    DATASTORES_BOUNDARY_EXAMPLES,
    check_saved_graph,
    get_test_mesh_dist,
    init_datastore_boundary_example,
    init_datastore_example,
)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize(
    "datastore_boundary_name",
    list(DATASTORES_BOUNDARY_EXAMPLES.keys()) + [None],
)
@pytest.mark.parametrize("archetype", ["keisler", "graphcast", "hierarchical"])
def test_build_archetype(datastore_name, datastore_boundary_name, archetype):
    """Check that the `build_graph_from_archetype` function is implemented.
    And that the graph is created in the correct location.

    """
    datastore = init_datastore_example(datastore_name)

    if datastore_boundary_name is None:
        # LAM scale
        datastore_boundary = None
    else:
        # Global scale, ERA5 coords flattened with proj
        datastore_boundary = init_datastore_boundary_example(
            datastore_boundary_name
        )

    create_kwargs = {
        "mesh_node_distance": get_test_mesh_dist(datastore, datastore_boundary),
    }

    num_levels = 1
    if archetype != "keisler":
        # Add additional multi-level kwargs
        create_kwargs.update(
            {
                "level_refinement_factor": 2,
                "max_num_levels": num_levels,
            }
        )

    # Name graph
    graph_name = f"{datastore_name}_{datastore_boundary_name}_{archetype}"

    # Saved in datastore
    # TODO: Maybe save in tmp dir?
    graph_dir_path = Path(datastore.root_path) / "graphs" / graph_name

    build_graph_from_archetype(
        datastore, datastore_boundary, graph_name, archetype, **create_kwargs
    )

    hierarchical = archetype == "hierarchical"
    check_saved_graph(graph_dir_path, hierarchical)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize(
    "datastore_boundary_name",
    list(DATASTORES_BOUNDARY_EXAMPLES.keys()) + [None],
)
@pytest.mark.parametrize(
    "config_i, graph_kwargs",
    enumerate(
        [
            # Assortment of options
            {
                "m2m_connectivity": "flat",
                "m2g_connectivity": "nearest_neighbour",
                "g2m_connectivity": "nearest_neighbour",
                "m2m_connectivity_kwargs": {},
            },
            {
                "m2m_connectivity": "flat_multiscale",
                "m2g_connectivity": "nearest_neighbours",
                "g2m_connectivity": "within_radius",
                "m2m_connectivity_kwargs": {
                    "level_refinement_factor": 2,
                },
                "m2g_connectivity_kwargs": {
                    "max_num_neighbours": 4,
                },
                "g2m_connectivity_kwargs": {
                    "rel_max_dist": 0.3,
                },
            },
            {
                "m2m_connectivity": "hierarchical",
                "m2g_connectivity": "containing_rectangle",
                "g2m_connectivity": "within_radius",
                "m2m_connectivity_kwargs": {
                    "level_refinement_factor": 2,
                },
                "m2g_connectivity_kwargs": {},
                "g2m_connectivity_kwargs": {
                    "rel_max_dist": 0.51,
                },
            },
        ]
    ),
)
def test_build_from_options(
    datastore_name, datastore_boundary_name, config_i, graph_kwargs
):
    """Check that the `build_graph_from_archetype` function is implemented.
    And that the graph is created in the correct location.

    """
    datastore = init_datastore_example(datastore_name)

    if datastore_boundary_name is None:
        # LAM scale
        datastore_boundary = None
    else:
        # Global scale, ERA5 coords flattened with proj
        datastore_boundary = init_datastore_boundary_example(
            datastore_boundary_name
        )

    # Insert mesh distance
    graph_kwargs["m2m_connectivity_kwargs"][
        "mesh_node_distance"
    ] = get_test_mesh_dist(datastore, datastore_boundary)

    # Name graph
    graph_name = f"{datastore_name}_{datastore_boundary_name}_config{config_i}"

    # Saved in datastore
    # TODO: Maybe save in tmp dir?
    graph_dir_path = Path(datastore.root_path) / "graphs" / graph_name

    build_graph(datastore, datastore_boundary, graph_name, **graph_kwargs)

    hierarchical = graph_kwargs["m2m_connectivity"] == "hierarchical"
    check_saved_graph(graph_dir_path, hierarchical)
