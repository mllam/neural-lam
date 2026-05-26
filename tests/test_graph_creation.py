# Standard library
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.utils import load_graph
from tests.conftest import init_datastore_example


@pytest.mark.parametrize("graph_name", ["1level", "multiscale", "hierarchical"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_graph_creation(datastore_name, graph_name):
    """Check that the `create_ graph_from_datastore` function is implemented.

    And that the graph is created in the correct location.

    """
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    elif graph_name == "multiscale":
        hierarchical = False
        n_max_levels = 3
    elif graph_name == "1level":
        hierarchical = False
        n_max_levels = 1
    else:
        raise ValueError(f"Unknown graph_name: {graph_name}")

    required_graph_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]
    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )

    # TODO: check that the number of edges is consistent over the files, for
    # now we just check the number of features
    d_features = 3
    d_mesh_static = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / graph_name

        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )

        assert graph_dir_path.exists()

        # check that all the required files are present
        for file_name in required_graph_files:
            assert (graph_dir_path / file_name).exists()

        # try to load each and ensure they have the right shape
        for file_name in required_graph_files:
            file_id = Path(file_name).stem  # remove the extension
            result = torch.load(graph_dir_path / file_name, weights_only=True)

            if file_id.startswith("g2m") or file_id.startswith("m2g"):
                assert isinstance(result, torch.Tensor)

                if file_id.endswith("_index"):
                    assert (
                        result.shape[0] == 2
                    )  # adjacency matrix uses two rows
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
                        assert len(result) == n_max_levels - 1
                    else:
                        assert len(result) == n_max_levels

                for r in result:
                    assert isinstance(r, torch.Tensor)

                    if file_id == "mesh_features":
                        assert r.shape[1] == d_mesh_static
                    elif file_id.endswith("_index"):
                        assert r.shape[0] == 2  # adjacency matrix uses two rows
                    elif file_id.endswith("_features"):
                        assert r.shape[1] == d_features


@pytest.mark.parametrize("graph_name", ["1level", "multiscale", "hierarchical"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_loaded_g2m_m2g_indices_in_bounds(datastore_name, graph_name):
    """Regression test for the hierarchical g2m/m2g zero-indexing bug.

    ``create_graph`` numbers grid nodes after the *total* mesh-node count
    across all hierarchy levels, so ``zero_index_g2m``/``zero_index_m2g``
    must subtract that same total. Previously they used only the
    bottom-level mesh count, which left grid indices offset out of bounds
    for hierarchical graphs and raised an ``IndexError`` in the GNN layers.

    Assert that, after ``load_graph`` zero-indexes the edges, the grid-side
    indices are zero-based and local (``< num_grid_points``) and the
    mesh-side indices are local (``< num_mesh_nodes``). With the
    bottom-level-only offset the grid-side indices stay shifted past
    ``num_grid_points`` for hierarchical graphs.
    """
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    elif graph_name == "multiscale":
        hierarchical = False
        n_max_levels = 3
    else:
        hierarchical = False
        n_max_levels = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / graph_name
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )

        is_hierarchical, graph = load_graph(graph_dir_path=graph_dir_path)

        mesh_static_features = graph["mesh_static_features"]
        if is_hierarchical:
            num_mesh_nodes = sum(sf.shape[0] for sf in mesh_static_features)
        else:
            num_mesh_nodes = mesh_static_features.shape[0]
        num_grid_nodes = datastore.num_grid_points

        # create_graph builds g2m edges grid -> mesh and m2g edges
        # mesh -> grid, so after zero-indexing:
        #   g2m_edge_index = [grid (sender), mesh (receiver)]
        #   m2g_edge_index = [mesh (sender), grid (receiver)]
        index_roles = {
            "g2m_edge_index": ("grid", "mesh"),
            "m2g_edge_index": ("mesh", "grid"),
        }
        bound = {"grid": num_grid_nodes, "mesh": num_mesh_nodes}

        for name, roles in index_roles.items():
            edge_index = graph[name]
            assert edge_index.min() >= 0, f"Negative index in {name}"
            for row, role in enumerate(roles):
                row_max = int(edge_index[row].max())
                assert row_max < bound[role], (
                    f"{name} {role}-side index {row_max} is out of bounds: "
                    f"expected a zero-based local index < {bound[role]} "
                    f"({role} node count). num_mesh_nodes={num_mesh_nodes}, "
                    f"num_grid_nodes={num_grid_nodes}."
                )
