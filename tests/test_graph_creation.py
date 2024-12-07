# Standard library
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam.build_rectangular_graph import (
    build_graph,
    build_graph_from_archetype,
)
from neural_lam.datastore import DATASTORES
from tests.conftest import (
    DATASTORES_BOUNDARY_EXAMPLES,
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
def test_graph_creation(datastore_name, datastore_boundary_name, archetype):
    """Check that the `create_ graph_from_datastore` function is implemented.
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
                "level_refinement_factor": 3,
                "max_num_levels": num_levels,
            }
        )

    required_graph_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "m2m_node_features.pt",
    ]

    hierarchical = archetype == "hierarchical"
    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )
        num_levels = 3

    # TODO: check that the number of edges is consistent over the files, for
    # now we just check the number of features
    d_features = 3
    d_mesh_static = 2

    # Name graph
    graph_name = f"{datastore_name}_{datastore_boundary_name}_{archetype}"

    # Saved in datastore
    # TODO: Maybe save in tmp dir?
    graph_dir_path = Path(datastore.root_path) / "graphs" / graph_name

    build_graph_from_archetype(
        datastore, datastore_boundary, graph_name, archetype, **create_kwargs
    )

    assert graph_dir_path.exists()

    # check that all the required files are present
    for file_name in required_graph_files:
        assert (graph_dir_path / file_name).exists()

    # try to load each and ensure they have the right shape
    for file_name in required_graph_files:
        file_id = Path(file_name).stem  # remove the extension
        result = torch.load(graph_dir_path / file_name)

        if file_id.startswith("g2m") or file_id.startswith("m2g"):
            assert isinstance(result, torch.Tensor)

            if file_id.endswith("_index"):
                assert result.shape[0] == 2  # adjacency matrix uses two rows
            elif file_id.endswith("_features"):
                assert result.shape[1] == d_features

        elif file_id.startswith("m2m") or file_id.startswith("mesh"):
            assert isinstance(result, list)
            if not hierarchical:
                assert len(result) == 1
            else:
                if file_id.startswith("mesh_up") or file_id.startswith(
                    "mesh_down"
                ):
                    assert len(result) == num_levels - 1
                else:
                    assert len(result) == num_levels

            for r in result:
                assert isinstance(r, torch.Tensor)

                if file_id == "m2m_node_features":
                    assert r.shape[1] == d_mesh_static
                elif file_id.endswith("_index"):
                    assert r.shape[0] == 2  # adjacency matrix uses two rows
                elif file_id.endswith("_features"):
                    assert r.shape[1] == d_features
