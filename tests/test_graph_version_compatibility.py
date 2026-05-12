# Standard library
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import utils
from neural_lam.create_graph import (
    GRAPH_NEURAL_LAM_VERSION_FILENAME,
    create_graph_from_datastore,
)
from tests.dummy_datastore import DummyDatastore


def _normalize_mesh_features(
    mesh_features: list[torch.Tensor],
) -> list[torch.Tensor]:
    normalized = [m.clone() for m in mesh_features]
    num_features = normalized[0].shape[1]

    pos_max = max(torch.max(torch.abs(m[:, :2])) for m in normalized)

    for feature_index in range(num_features):
        if feature_index < 2:
            scale = pos_max
        else:
            scale = max(
                torch.max(torch.abs(m[:, feature_index])) for m in normalized
            )
            if scale == 0:
                scale = 1.0

        for mesh_tensor in normalized:
            mesh_tensor[:, feature_index] /= scale

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

        raw_mesh_features = torch.load(graph_dir_path / "mesh_features.pt")
        expected_mesh_features = _normalize_mesh_features(raw_mesh_features)

        # New-format graphs should normalize mesh node coordinates on load.
        assert not torch.allclose(
            raw_mesh_features[0], expected_mesh_features[0]
        )

        _, graph_ldict = utils.load_graph(graph_dir_path=str(graph_dir_path))
        assert torch.allclose(
            graph_ldict["mesh_static_features"], expected_mesh_features[0]
        )

        # Deleting the sentinel makes the graph look legacy, so load_graph()
        # should leave the mesh node features exactly as they were stored
        # on disk.
        (graph_dir_path / GRAPH_NEURAL_LAM_VERSION_FILENAME).unlink()

        with pytest.warns(RuntimeWarning, match="neural-lam<=0.6.0"):
            _, legacy_graph_ldict = utils.load_graph(
                graph_dir_path=str(graph_dir_path)
            )

        assert torch.allclose(
            legacy_graph_ldict["mesh_static_features"], raw_mesh_features[0]
        )
