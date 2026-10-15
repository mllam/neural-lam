# Standard library
import copy
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import utils
from neural_lam.create_graph import create_graph_from_datastore
from tests.dummy_datastore import DummyDatastore


def _normalize_mesh_features(
    mesh_features: list[torch.Tensor], grid_xy_max_span: float
) -> list[torch.Tensor]:
    normalized = [m.clone() for m in mesh_features]

    for mesh_tensor in normalized:
        mesh_tensor[:, :2] /= grid_xy_max_span

    return normalized


def _load_edge_index(path: Path) -> torch.Tensor | list[torch.Tensor]:
    return torch.load(path)


def _write_legacy_graph(
    graph_dir_path: Path,
    mesh_features: list[torch.Tensor],
    m2m_edge_index: list[torch.Tensor],
    m2m_features: list[torch.Tensor],
    g2m_edge_index: torch.Tensor,
    m2g_edge_index: torch.Tensor,
    num_grid_nodes: int,
    num_mesh_nodes: int,
) -> None:
    """Write a legacy-format graph with combined-offset edge indices."""
    # Legacy format: mesh nodes first, then grid nodes.
    # Edge indices use combined offset: grid indices shifted by num_mesh_nodes.
    legacy_g2m = g2m_edge_index.clone()
    legacy_g2m[0] += num_mesh_nodes  # grid senders offset by mesh node count
    legacy_m2g = m2g_edge_index.clone()
    legacy_m2g[1] += num_mesh_nodes  # grid receivers offset by mesh node count

    torch.save(mesh_features, graph_dir_path / "mesh_features.pt")
    torch.save(m2m_edge_index, graph_dir_path / "m2m_edge_index.pt")
    torch.save(m2m_features, graph_dir_path / "m2m_features.pt")
    torch.save(legacy_g2m, graph_dir_path / "g2m_edge_index.pt")
    torch.save(legacy_m2g, graph_dir_path / "m2g_edge_index.pt")
    torch.save(
        torch.zeros((g2m_edge_index.shape[1], 3), dtype=torch.float32),
        graph_dir_path / "g2m_features.pt",
    )
    torch.save(
        torch.zeros((m2g_edge_index.shape[1], 3), dtype=torch.float32),
        graph_dir_path / "m2g_features.pt",
    )


def test_load_graph_respects_current_and_legacy_mesh_feature_formats():
    """Check current graphs normalize mesh node features.

    Also check legacy graphs keep mesh node features unnormalized.
    """

    datastore = DummyDatastore()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "1level"
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=False,
            n_max_levels=1,
        )

        grid_xy_extent = datastore.get_xy_extent(category="state")
        grid_xy_max_span = max(
            grid_xy_extent[1] - grid_xy_extent[0],
            grid_xy_extent[3] - grid_xy_extent[2],
        )

        raw_mesh_features = torch.load(graph_dir_path / "mesh_features.pt")
        raw_m2m_edge_index = _load_edge_index(
            graph_dir_path / "m2m_edge_index.pt"
        )
        raw_g2m_edge_index = _load_edge_index(
            graph_dir_path / "g2m_edge_index.pt"
        )
        raw_m2g_edge_index = _load_edge_index(
            graph_dir_path / "m2g_edge_index.pt"
        )
        raw_m2m_features = _load_edge_index(graph_dir_path / "m2m_features.pt")
        expected_mesh_features = _normalize_mesh_features(
            raw_mesh_features, grid_xy_max_span
        )

        # New-format graphs should normalize mesh node coordinates on load.
        assert not torch.allclose(
            raw_mesh_features[0], expected_mesh_features[0]
        )

        _, graph_ldict = utils.load_graph(
            graph_dir_path=str(graph_dir_path),
            mesh_node_features_scaling=grid_xy_max_span,
        )
        assert torch.allclose(
            graph_ldict["mesh_static_features"], expected_mesh_features[0]
        )
        assert torch.equal(graph_ldict["g2m_edge_index"], raw_g2m_edge_index)
        assert torch.equal(graph_ldict["m2g_edge_index"], raw_m2g_edge_index)
        assert torch.equal(graph_ldict["m2m_edge_index"], raw_m2m_edge_index[0])

        # Build a legacy-format graph directory and verify it loads without
        # normalization and with zero-offset edge indices.
        num_mesh_nodes = raw_mesh_features[0].shape[0]
        grid_xy = datastore.get_xy(category="state", stacked=False)
        num_grid_nodes = grid_xy.shape[0] * grid_xy.shape[1]

        legacy_dir = Path(tmpdir) / "graph" / "legacy"
        legacy_dir.mkdir(parents=True)
        _write_legacy_graph(
            legacy_dir,
            mesh_features=copy.deepcopy(raw_mesh_features),
            m2m_edge_index=copy.deepcopy(raw_m2m_edge_index),
            g2m_edge_index=copy.deepcopy(raw_g2m_edge_index),
            m2g_edge_index=copy.deepcopy(raw_m2g_edge_index),
            m2m_features=copy.deepcopy(raw_m2m_features),
            num_grid_nodes=num_grid_nodes,
            num_mesh_nodes=num_mesh_nodes,
        )

        with pytest.warns(RuntimeWarning, match="legacy pre-spec format"):
            _, legacy_graph_ldict = utils.load_graph(
                graph_dir_path=str(legacy_dir),
                mesh_node_features_scaling=grid_xy_max_span,
            )

        # Legacy: mesh features left as-is (not normalized)
        assert torch.allclose(
            legacy_graph_ldict["mesh_static_features"], raw_mesh_features[0]
        )
        # Legacy: edge indices zero-offset on load
        assert torch.equal(
            legacy_graph_ldict["g2m_edge_index"], raw_g2m_edge_index
        )
        assert torch.equal(
            legacy_graph_ldict["m2g_edge_index"], raw_m2g_edge_index
        )
