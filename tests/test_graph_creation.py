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
    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )

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
            result = torch.load(graph_dir_path / file_name)

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

        # Ensure edge counts match between *_edge_index.pt and *_features.pt
        g2m_ei = torch.load(graph_dir_path / "g2m_edge_index.pt")
        g2m_f = torch.load(graph_dir_path / "g2m_features.pt")
        assert g2m_ei.shape[1] == g2m_f.shape[0], "g2m edge count mismatch"

        m2g_ei = torch.load(graph_dir_path / "m2g_edge_index.pt")
        m2g_f = torch.load(graph_dir_path / "m2g_features.pt")
        assert m2g_ei.shape[1] == m2g_f.shape[0], "m2g edge count mismatch"

        m2m_ei = torch.load(graph_dir_path / "m2m_edge_index.pt")
        m2m_f = torch.load(graph_dir_path / "m2m_features.pt")
        assert len(m2m_ei) == len(m2m_f)
        for i, (ei, f) in enumerate(zip(m2m_ei, m2m_f)):
            assert ei.shape[1] == f.shape[0], f"m2m level {i} edge count mismatch"

        if hierarchical:
            up_ei = torch.load(graph_dir_path / "mesh_up_edge_index.pt")
            up_f = torch.load(graph_dir_path / "mesh_up_features.pt")
            assert len(up_ei) == len(up_f)
            for i, (ei, f) in enumerate(zip(up_ei, up_f)):
                assert ei.shape[1] == f.shape[0], (
                    f"mesh_up level {i} edge count mismatch"
                )

            down_ei = torch.load(graph_dir_path / "mesh_down_edge_index.pt")
            down_f = torch.load(graph_dir_path / "mesh_down_features.pt")
            assert len(down_ei) == len(down_f)
            for i, (ei, f) in enumerate(zip(down_ei, down_f)):
                assert ei.shape[1] == f.shape[0], (
                    f"mesh_down level {i} edge count mismatch"
                )
