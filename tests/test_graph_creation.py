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


def _ctx(datastore_name: str, graph_name: str, pair: str) -> str:
    """Short prefix for assertion messages (CI-friendly)."""
    return f"datastore={datastore_name!r}, graph={graph_name!r}, pair={pair}"


def _assert_tensor_edge_feature_agree(
    edge_index,
    features,
    *,
    datastore_name: str,
    graph_name: str,
    pair: str,
    d_features: int,
) -> None:
    """One feature row per column in *edge_index* (one edge); validate types/shapes."""
    c = _ctx(datastore_name, graph_name, pair)
    assert isinstance(edge_index, torch.Tensor), (
        f"{c}: edge_index must be torch.Tensor, got {type(edge_index).__name__}"
    )
    assert isinstance(features, torch.Tensor), (
        f"{c}: features must be torch.Tensor, got {type(features).__name__}"
    )
    assert edge_index.dim() == 2, (
        f"{c}: edge_index must be 2D, got dim={edge_index.dim()} "
        f"shape={tuple(edge_index.shape)}"
    )
    assert edge_index.shape[0] == 2, (
        f"{c}: edge_index must have 2 rows (COO), got shape[0]={edge_index.shape[0]}"
    )
    assert features.dim() == 2, (
        f"{c}: features must be 2D, got dim={features.dim()} "
        f"shape={tuple(features.shape)}"
    )
    assert features.shape[1] == d_features, (
        f"{c}: features last dim must be d_features={d_features}, "
        f"got {features.shape[1]}"
    )
    n_edges = edge_index.shape[1]
    n_rows = features.shape[0]
    assert n_edges == n_rows, (
        f"{c}: edge_index/features row mismatch — edge_index has {n_edges} edges "
        f"(shape[1]={n_edges}) but features has {n_rows} rows (shape[0]={n_rows})"
    )


def _assert_list_edge_feature_agree(
    edge_indices,
    features_list,
    *,
    datastore_name: str,
    graph_name: str,
    pair: str,
    d_features: int,
) -> None:
    """Same as :func:`_assert_tensor_edge_feature_agree` but for per-level lists."""
    c = _ctx(datastore_name, graph_name, pair)
    assert isinstance(edge_indices, list), (
        f"{c}: edge_index artifact must be a list, got {type(edge_indices).__name__}"
    )
    assert isinstance(features_list, list), (
        f"{c}: features artifact must be a list, got {type(features_list).__name__}"
    )
    assert len(edge_indices) == len(features_list), (
        f"{c}: list length mismatch: {len(edge_indices)} edge_index tensors vs "
        f"{len(features_list)} feature tensors"
    )
    for level, (ei, ft) in enumerate(zip(edge_indices, features_list)):
        _assert_tensor_edge_feature_agree(
            ei,
            ft,
            datastore_name=datastore_name,
            graph_name=graph_name,
            pair=f"{pair}[level={level}]",
            d_features=d_features,
        )


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

        g2m_ei = torch.load(graph_dir_path / "g2m_edge_index.pt")
        g2m_ft = torch.load(graph_dir_path / "g2m_features.pt")
        _assert_tensor_edge_feature_agree(
            g2m_ei,
            g2m_ft,
            datastore_name=datastore_name,
            graph_name=graph_name,
            pair="g2m",
            d_features=d_features,
        )

        m2g_ei = torch.load(graph_dir_path / "m2g_edge_index.pt")
        m2g_ft = torch.load(graph_dir_path / "m2g_features.pt")
        _assert_tensor_edge_feature_agree(
            m2g_ei,
            m2g_ft,
            datastore_name=datastore_name,
            graph_name=graph_name,
            pair="m2g",
            d_features=d_features,
        )

        m2m_ei = torch.load(graph_dir_path / "m2m_edge_index.pt")
        m2m_ft = torch.load(graph_dir_path / "m2m_features.pt")
        _assert_list_edge_feature_agree(
            m2m_ei,
            m2m_ft,
            datastore_name=datastore_name,
            graph_name=graph_name,
            pair="m2m",
            d_features=d_features,
        )

        if hierarchical:
            up_ei = torch.load(graph_dir_path / "mesh_up_edge_index.pt")
            up_ft = torch.load(graph_dir_path / "mesh_up_features.pt")
            _assert_list_edge_feature_agree(
                up_ei,
                up_ft,
                datastore_name=datastore_name,
                graph_name=graph_name,
                pair="mesh_up",
                d_features=d_features,
            )

            down_ei = torch.load(graph_dir_path / "mesh_down_edge_index.pt")
            down_ft = torch.load(graph_dir_path / "mesh_down_features.pt")
            _assert_list_edge_feature_agree(
                down_ei,
                down_ft,
                datastore_name=datastore_name,
                graph_name=graph_name,
                pair="mesh_down",
                d_features=d_features,
            )
