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

    # index-feature pair to check if edge is consistent across files
    edge_index_feature_pairs = [
        ("g2m_edge_index", "g2m_features"),
        ("m2g_edge_index", "m2g_features"),
        ("m2m_edge_index", "m2m_features"),
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
        edge_index_feature_pairs.extend(
            [
                ("mesh_up_edge_index", "mesh_up_features"),
                ("mesh_down_edge_index", "mesh_down_features"),
            ]
        )

    # check that the number of edges is consistent over the files
    d_features = 3
    d_mesh_static = 2
    edge_counts = {}

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
            result = torch.load(graph_dir_path / file_name)
            if file_id.startswith("g2m") or file_id.startswith("m2g"):
                assert isinstance(result, torch.Tensor)
                if file_id.endswith("_index"):
                    assert (
                        result.shape[0] == 2
                    )  # adjacency matrix uses two rows
                    edge_counts[file_id] = result.shape[1]
                elif file_id.endswith("_features"):
                    assert result.shape[1] == d_features
                    edge_counts[file_id] = result.shape[0]

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

                if file_id.endswith("_index"):
                    edge_counts[file_id] = [r.shape[1] for r in result]
                elif (
                    file_id.endswith("_features") and file_id != "mesh_features"
                ):
                    edge_counts[file_id] = [r.shape[0] for r in result]

    # loop through index-feature pair to check consistency
    for index_id, features_id in edge_index_feature_pairs:
        assert edge_counts[index_id] == edge_counts[features_id], (
            f"Edge count mismatch: {index_id} has {edge_counts[index_id]} edges"
            f" but {features_id} has {edge_counts[features_id]} rows"
        )
