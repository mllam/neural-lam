# Standard library
import importlib.util
import tempfile
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam.create_graph import (
    CURRENT_GRAPH_SPEC_VERSION,
    METAINFO_FILENAME,
    create_graph_from_datastore,
)
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from tests.conftest import init_datastore_example


def _load_validator_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "docs" / "validate_graph.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_graph_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    # Standard library
    import sys

    sys.modules["validate_graph_script"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("graph_name", ["1level", "multiscale", "hierarchical"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_graph_creation(datastore_name, graph_name):
    """Check that the `create_ graph_from_datastore` function is implemented.

    And that the graph is created in the correct location.

    """
    validator = _load_validator_module()
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular grid datastore."  # noqa: E501
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
        METAINFO_FILENAME,
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

        # Third-party
        import yaml

        meta = yaml.safe_load(
            (graph_dir_path / METAINFO_FILENAME).read_text(encoding="utf-8")
        )
        assert meta is not None
        assert meta["spec_version"] == CURRENT_GRAPH_SPEC_VERSION

        report, _, _ = validator.validate_graph_directory(graph_dir_path)
        assert not report.has_fails(), report.summarize()

        # try to load each and ensure they have the right shape
        for file_name in required_graph_files:
            if file_name == METAINFO_FILENAME:
                continue
            file_id = Path(file_name).stem  # remove the extension
            result = torch.load(graph_dir_path / file_name, weights_only=True)

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
