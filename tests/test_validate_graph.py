import tempfile
from pathlib import Path

import pytest
import torch

from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.graph_validation import validate_graph_dir
from tests.conftest import init_datastore_example


def test_validate_graph_dir_ok():
    datastore = init_datastore_example("dummydata")
    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("requires regular grid datastore")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir = Path(tmpdir) / "graph" / "1level"
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir),
            hierarchical=False,
            n_max_levels=1,
        )
        errors = validate_graph_dir(
            graph_dir,
            expected_hierarchical=False,
            expected_n_levels=1,
            expected_d_edge_features=3,
            expected_d_mesh_features=2,
        )
        assert errors == [], "\n".join(e.format() for e in errors)


def test_validate_graph_dir_reports_edge_count_mismatch():
    datastore = init_datastore_example("dummydata")
    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("requires regular grid datastore")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir = Path(tmpdir) / "graph" / "1level"
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir),
            hierarchical=False,
            n_max_levels=1,
        )

        # Corrupt g2m_features by truncating one row
        g2m_features_path = graph_dir / "g2m_features.pt"
        feats = torch.load(g2m_features_path, weights_only=True)
        torch.save(feats[:-1], g2m_features_path)

        errors = validate_graph_dir(graph_dir)
        formatted = "\n".join(e.format() for e in errors)
        assert "g2m" in formatted.lower()
        assert "mismatch" in formatted.lower()

