# Standard library
import tempfile
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch

# First-party
from neural_lam.create_graph import create_graph, create_graph_from_datastore
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


def test_graph_creation_non_square_aspect_ratio():
    """
    Mesh at level 1 should reflect the domain's aspect ratio, not be square.
    """
    Nx, Ny = 100, 600
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 6, Ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([xx, yy], axis=-1)  # (100, 600, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        create_graph(graph_dir_path=tmpdir, xy=xy, n_max_levels=1)
        mesh_features = torch.load(f"{tmpdir}/mesh_features.pt")
        n_mesh_nodes = mesh_features[0].shape[0]

    # With a square mesh, n_mesh_nodes would be n x n.
    # With correct aspect ratio, n_y > n_x, so nodes < n_larger^2.
    n_larger = int(3 ** int(np.log(max(Nx, Ny)) / np.log(3)) / 3)
    assert n_mesh_nodes < n_larger**2


def test_graph_creation_multiscale_non_square():
    """
    Multi-level rectangular mesh must not crash at reshape.
    Mesh should also reflect the domain aspect ratio.
    """
    Nx, Ny = 100, 600
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 6, Ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([xx, yy], axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        create_graph(graph_dir_path=tmpdir, xy=xy, n_max_levels=2)
        mesh_features = torch.load(f"{tmpdir}/mesh_features.pt")
        n_mesh_nodes = mesh_features[0].shape[0]

    n_larger = int(3 ** int(np.log(max(Nx, Ny)) / np.log(3)) / 3)
    assert n_mesh_nodes < n_larger**2


def test_graph_creation_square_g2m_edges_unchanged():
    """
    Square grid g2m edge count must be preserved.
    Edge count must use the original square formula over rectangular domains.
    """
    Nx, Ny = 81, 81
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([xx, yy], axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        create_graph(graph_dir_path=tmpdir, xy=xy, n_max_levels=1)
        g2m = torch.load(f"{tmpdir}/g2m_edge_index.pt")
        assert (
            g2m.shape[1] == 9009
        ), f"Square grid g2m edges changed: expected 9009, got {g2m.shape[1]}"
