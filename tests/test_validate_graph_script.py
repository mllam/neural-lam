# Standard library
import importlib.util
import tempfile
from pathlib import Path

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from tests.conftest import init_datastore_example


def _load_validator_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "docs" / "validate_graph_new.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_graph_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_graph_script_for_all_graph_types():
    validator = _load_validator_module()
    datastore = init_datastore_example("dummydata")

    cases = [
        ("1level", False, 1, 1),
        ("multiscale", False, 3, 1),
        ("hierarchical", True, 3, 3),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, hierarchical, n_max_levels, expected_levels in cases:
            graph_dir_path = Path(tmpdir) / "graph" / name
            create_graph_from_datastore(
                datastore=datastore,
                output_root_path=str(graph_dir_path),
                hierarchical=hierarchical,
                n_max_levels=n_max_levels,
            )

            report, spec, props = validator.validate_graph_directory(
                graph_dir_path
            )
            assert not report.has_fails(), f"{name} failed validation"
            assert len(props.num_mesh_nodes_per_level) == expected_levels
            assert props.hierarchical == hierarchical
