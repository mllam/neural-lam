# Standard library
import importlib.util
import tempfile
from pathlib import Path

# Third-party
import torch


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


def _edge_features(edge_index: torch.Tensor) -> torch.Tensor:
    num_edges = edge_index.shape[1]
    return torch.zeros((num_edges, 3), dtype=torch.float32)


def _write_graph(
    graph_dir_path: Path,
    *,
    hierarchical: bool = False,
    version: str | None = "0.1.0",
) -> None:
    graph_dir_path.mkdir(parents=True)

    if version is not None:
        (graph_dir_path / "graph-spec-version").write_text(
            version, encoding="utf-8"
        )

    mesh_features = [
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ]
    m2m_edge_index = [torch.tensor([[0, 1, 2], [1, 2, 0]])]
    m2m_features = [_edge_features(m2m_edge_index[0])]

    if hierarchical:
        mesh_features.append(
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        )
        m2m_edge_index.append(torch.tensor([[0, 1], [1, 0]]))
        m2m_features.append(_edge_features(m2m_edge_index[1]))

        mesh_up_edge_index = [torch.tensor([[0, 1, 2], [0, 1, 0]])]
        mesh_down_edge_index = [torch.tensor([[0, 1, 0], [0, 1, 2]])]
        torch.save(mesh_up_edge_index, graph_dir_path / "mesh_up_edge_index.pt")
        torch.save(
            mesh_down_edge_index, graph_dir_path / "mesh_down_edge_index.pt"
        )
        torch.save(
            [_edge_features(mesh_up_edge_index[0])],
            graph_dir_path / "mesh_up_features.pt",
        )
        torch.save(
            [_edge_features(mesh_down_edge_index[0])],
            graph_dir_path / "mesh_down_features.pt",
        )

    g2m_edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    m2g_edge_index = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]])

    torch.save(mesh_features, graph_dir_path / "mesh_features.pt")
    torch.save(m2m_edge_index, graph_dir_path / "m2m_edge_index.pt")
    torch.save(m2m_features, graph_dir_path / "m2m_features.pt")
    torch.save(g2m_edge_index, graph_dir_path / "g2m_edge_index.pt")
    torch.save(m2g_edge_index, graph_dir_path / "m2g_edge_index.pt")
    torch.save(
        _edge_features(g2m_edge_index), graph_dir_path / "g2m_features.pt"
    )
    torch.save(
        _edge_features(m2g_edge_index), graph_dir_path / "m2g_features.pt"
    )


def _details(report) -> list[str]:
    return [result.detail for result in report.results]


def _warnings(report):
    return [result for result in report.results if result.status == "WARNING"]


def test_validate_graph_script_for_flat_and_hierarchical_graphs():
    validator = _load_validator_module()

    cases = [("flat", False, 1), ("hierarchical", True, 2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, is_hierarchical, expected_levels in cases:
            graph_dir_path = Path(tmpdir) / "graph" / name
            _write_graph(graph_dir_path, hierarchical=is_hierarchical)

            report, spec, props = validator.validate_graph_directory(
                graph_dir_path
            )
            assert not report.has_fails(), f"{name} failed validation"
            assert len(props.num_mesh_nodes_per_level) == expected_levels
            assert props.is_hierarchical == is_hierarchical


def test_validate_graph_script_rejects_g2m_receiver_outside_bottom_mesh_range():
    validator = _load_validator_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "bad-g2m-receiver"
        _write_graph(graph_dir_path)
        torch.save(
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            graph_dir_path / "g2m_edge_index.pt",
        )

        report, _, _ = validator.validate_graph_directory(graph_dir_path)

    assert report.has_fails()
    assert any(
        "g2m_edge_index: receiver indices out" in d for d in _details(report)
    )


def test_validate_graph_script_rejects_hierarchical_offset_indices():
    validator = _load_validator_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "offset-hierarchical"
        _write_graph(graph_dir_path, hierarchical=True)
        torch.save(
            [torch.tensor([[0, 1, 2], [1, 2, 0]]), torch.tensor([[3], [4]])],
            graph_dir_path / "m2m_edge_index.pt",
        )

        report, _, _ = validator.validate_graph_directory(graph_dir_path)

    assert report.has_fails()
    assert not report.ok
    assert any(
        "m2m_edge_index[1]: sender indices out" in d for d in _details(report)
    )


def test_validate_graph_script_warns_when_grid_subset_does_not_start_at_zero():
    validator = _load_validator_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "grid-subset"
        _write_graph(graph_dir_path)
        torch.save(
            torch.tensor([[0, 1, 2, 0], [1, 2, 3, 1]]),
            graph_dir_path / "m2g_edge_index.pt",
        )

        report, _, _ = validator.validate_graph_directory(graph_dir_path)

    assert not report.has_fails()
    assert any(
        "m2g_edge_index row 1 has minimum grid index 1 rather than 0"
        in warning.detail
        and "not all grid nodes are decoded to" in warning.detail
        for warning in _warnings(report)
    )


def test_validate_graph_script_requires_graph_spec_version():
    validator = _load_validator_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "legacy"
        _write_graph(graph_dir_path, version=None)

        report, _, _ = validator.validate_graph_directory(graph_dir_path)

    assert report.has_fails()
    assert not report.ok
    assert any("graph-spec-version is missing" in d for d in _details(report))


def test_validate_graph_script_rejects_unsupported_graph_spec_version():
    validator = _load_validator_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / "unsupported"
        _write_graph(graph_dir_path, version="9.9.9")

        report, _, props = validator.validate_graph_directory(graph_dir_path)

    assert report.has_fails()
    assert not report.ok
    assert props.graph_format_spec_version == "9.9.9"
    assert any("unsupported spec version" in d for d in _details(report))
