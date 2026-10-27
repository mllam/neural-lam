"""Exercise metadata validation before any graph tensors are loaded."""

# Standard library
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

# Third-party
import pytest


@pytest.mark.parametrize(
    "metadata",
    [
        "[]",
        "0.1.0",
        "42",
        "true",
        "2026-01-01",
        "null",
        "{}",
        "spec_version: [",
    ],
)
def test_validator_reports_invalid_metadata_as_json(tmp_path, metadata):
    """Invalid metadata produces a failed report, not a traceback."""
    (tmp_path / "metainfo.yaml").write_text(metadata, encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "docs" / "validate_graph.py"

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert any(
        item["requirement"] == "Graph spec version"
        and item["status"] == "FAIL"
        and "missing or invalid 'spec_version'" in item["detail"]
        for item in report["results"]
    )
    assert "Traceback" not in result.stderr


def test_validator_accepts_mapping_metadata_before_loading_tensors(
    tmp_path, monkeypatch
):
    """A supported version passes metadata validation before tensor loading."""
    script = Path(__file__).resolve().parents[1] / "docs" / "validate_graph.py"
    spec = importlib.util.spec_from_file_location(
        "validate_graph_script", script
    )
    assert spec is not None
    validator = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, validator)
    assert spec.loader is not None
    spec.loader.exec_module(validator)

    (tmp_path / "metainfo.yaml").write_text(
        f"spec_version: '{validator.CURRENT_GRAPH_FORMAT_SPEC_VERSION}'\n",
        encoding="utf-8",
    )

    class TensorLoadingReached(Exception):
        """Stop after metadata validation, without needing graph tensors."""

    def stop_at_tensor_loading(path):
        assert path == tmp_path / "m2m_edge_index.pt"
        raise TensorLoadingReached

    monkeypatch.setattr(validator, "_load_pt", stop_at_tensor_loading)
    # Invalid metadata returns a failed report before reaching this point.
    with pytest.raises(TensorLoadingReached):
        validator.validate_graph_directory(tmp_path)
