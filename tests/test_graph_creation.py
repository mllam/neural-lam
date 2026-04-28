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


def _mesh_nodes_per_level(mesh_features: list) -> list[int]:
    """Number of mesh nodes at each level (matches ``mesh_features.pt`` layout)."""
    return [int(t.shape[0]) for t in mesh_features]


def _assert_edge_index_in_range(
    edge_index: torch.Tensor,
    *,
    n_nodes: int,
    name: str,
) -> None:
    """All entries must be valid node ids in ``[0, n_nodes)``."""
    assert edge_index.numel() > 0, f"{name}: empty edge_index"
    assert torch.all(edge_index >= 0), f"{name}: negative node index"
    assert torch.all(edge_index < n_nodes), (
        f"{name}: node index out of range (max={edge_index.max().item()}, "
        f"n_nodes={n_nodes})"
    )


def _assert_m2m_edge_indices_per_level(
    m2m_indices: list,
    mesh_features: list,
) -> None:
    """m2m edges at level ``l`` use global mesh ids for that level's slice."""
    sizes = _mesh_nodes_per_level(mesh_features)
    assert len(m2m_indices) == len(sizes), "m2m levels vs mesh_features mismatch"
    start = 0
    for level, (ei, n_l) in enumerate(zip(m2m_indices, sizes)):
        _assert_edge_index_in_range(
            ei,
            n_nodes=start + n_l,
            name=f"m2m_edge_index level {level}",
        )
        # Stronger check: indices stay inside this level's block.
        assert torch.all(ei >= start), (
            f"m2m level {level}: index below start {start} "
            f"(min={ei.min().item()})"
        )
        assert torch.all(ei < start + n_l), (
            f"m2m level {level}: index above block end {start + n_l - 1}"
        )
        start += n_l


def _assert_hierarchical_mesh_up_down_in_range(
    mesh_up: list,
    mesh_down: list,
    n_mesh_total: int,
) -> None:
    """Up/down edges only reference global mesh node ids."""
    assert len(mesh_up) == len(mesh_down)
    for i, (up_ei, down_ei) in enumerate(zip(mesh_up, mesh_down)):
        _assert_edge_index_in_range(
            up_ei,
            n_nodes=n_mesh_total,
            name=f"mesh_up_edge_index[{i}]",
        )
        _assert_edge_index_in_range(
            down_ei,
            n_nodes=n_mesh_total,
            name=f"mesh_down_edge_index[{i}]",
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

        # Edge indices must reference valid node ids (grid+mesh or mesh-only).
        xy = datastore.get_xy(category="state", stacked=False)
        n_grid = int(xy.shape[0] * xy.shape[1])

        mesh_pos_list = torch.load(graph_dir_path / "mesh_features.pt")
        n_mesh_total = sum(int(t.shape[0]) for t in mesh_pos_list)
        n_bipartite = n_grid + n_mesh_total

        g2m_ei = torch.load(graph_dir_path / "g2m_edge_index.pt")
        m2g_ei = torch.load(graph_dir_path / "m2g_edge_index.pt")
        m2m_ei_list = torch.load(graph_dir_path / "m2m_edge_index.pt")

        _assert_edge_index_in_range(
            g2m_ei, n_nodes=n_bipartite, name="g2m_edge_index"
        )
        _assert_edge_index_in_range(
            m2g_ei, n_nodes=n_bipartite, name="m2g_edge_index"
        )
        _assert_m2m_edge_indices_per_level(m2m_ei_list, mesh_pos_list)

        if hierarchical:
            mesh_up_list = torch.load(graph_dir_path / "mesh_up_edge_index.pt")
            mesh_down_list = torch.load(graph_dir_path / "mesh_down_edge_index.pt")
            _assert_hierarchical_mesh_up_down_in_range(
                mesh_up_list, mesh_down_list, n_mesh_total
            )
