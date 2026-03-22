import json
import tempfile
from pathlib import Path

import pytest

from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.graph_generation_config import GRAPH_CONFIG_FILENAME
from neural_lam.graph_validation import validate_graph_dir
from tests.conftest import init_datastore_example


def test_graph_config_file_written_with_expected_keys():
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
        cfg_path = graph_dir / GRAPH_CONFIG_FILENAME
        assert cfg_path.is_file()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["hierarchical"] is False
        assert data["n_levels_on_disk"] == 1
        assert "grid_nx" in data and "grid_ny" in data
        assert "neural_lam_version" in data
        assert "generated_at_utc" in data
        assert "mesh_tree" in data and data["mesh_tree"]["branching_children_per_axis"] == 3
        assert data["g2m"]["dm_scale"] == 0.67
        assert data["tensor_layout"]["d_edge_features"] == 3


def test_validate_graph_dir_passes_with_graph_config_present():
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
        assert errors == []


def test_validate_graph_dir_require_graph_config_when_missing():
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
        (graph_dir / GRAPH_CONFIG_FILENAME).unlink()
        errors = validate_graph_dir(graph_dir, require_graph_config=True)
        assert errors
        assert any("missing graph_config.json" in e.message for e in errors)


def test_validate_graph_dir_invalid_graph_config_json():
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
        (graph_dir / GRAPH_CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
        errors = validate_graph_dir(graph_dir)
        assert errors
        assert any("invalid JSON" in e.message for e in errors)
