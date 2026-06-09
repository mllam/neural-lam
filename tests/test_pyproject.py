# Standard library
import tomllib
from pathlib import Path


def test_slow_marker_registered_in_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("slow:") for marker in markers)
