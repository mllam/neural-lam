# Standard library
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import utils
from neural_lam.create_graph import (
    GRAPH_SPEC_VERSION_FILENAME,
    create_graph_from_datastore,
)
from tests.dummy_datastore import DummyDatastore


def _normalize_mesh_features(
    mesh_features: list[torch.Tensor], grid_xy_max_span: float
) -> list[torch.Tensor]:
    normalized = [m.clone() for m in mesh_features]

    for mesh_tensor in normalized:
        mesh_tensor[:, :2] /= grid_xy_max_span

    return normalized


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

        # Deleting the spec version file makes the graph look legacy, so
        # load_graph() should leave the mesh node features exactly as they
        # were stored on disk.
        (graph_dir_path / GRAPH_SPEC_VERSION_FILENAME).unlink()

        with pytest.warns(RuntimeWarning, match="legacy pre-spec format"):
            _, legacy_graph_ldict = utils.load_graph(
                graph_dir_path=str(graph_dir_path),
                mesh_node_features_scaling=grid_xy_max_span,
            )

        assert torch.allclose(
            legacy_graph_ldict["mesh_static_features"], raw_mesh_features[0]
        )
