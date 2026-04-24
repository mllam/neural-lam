"""
Standalone CLI validator for neural-lam on-disk graph directories.

Run with:
    uv run docs/validate_graph.py --graph_dir <path-to-graph-dir>
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.2",
#   "torch>=2.3.0",
# ]
# ///

# Standard library
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Third-party
import torch

EDGE_FEATURE_DIM = 3
MESH_FEATURE_DIM = 2


@dataclass
class GraphValidationReport:
    """
    Structured result of validating a graph directory.

    Attributes:
        graph_dir (str): Validated graph directory path.
        hierarchical (bool): Whether the graph has hierarchical mesh levels.
        num_levels (int): Number of mesh levels encoded in `m2m_*`/`mesh_*`.
        num_mesh_nodes_per_level (list[int]): Mesh node count for each level.
        num_mesh_nodes_total (int): Total number of mesh nodes across levels.
        num_grid_nodes (int): Inferred number of grid nodes.
        errors (list[str]): Fatal validation failures.
        warnings (list[str]): Non-fatal validation issues.
    """

    graph_dir: str
    hierarchical: bool
    num_levels: int
    num_mesh_nodes_per_level: list[int]
    num_mesh_nodes_total: int
    num_grid_nodes: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _load_pt(path: Path) -> Any:
    """
    Load a torch-serialized object from disk.

    Args:
        path (Path): Path to a `.pt` file.

    Returns:
        Any: Deserialized object from the `.pt` file.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether a tensor uses an integer dtype.

    Args:
        tensor (torch.Tensor): Tensor to check.

    Returns:
        bool: `True` if tensor dtype is integer, else `False`.
    """
    return tensor.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )


def _check_edge_index(
    *,
    name: str,
    edge_index: torch.Tensor,
    errors: list[str],
    expected_sender_range: tuple[int, int] | None = None,
    expected_receiver_range: tuple[int, int] | None = None,
) -> int:
    """
    Validate edge index tensor shape, dtype, and optional index ranges.

    Args:
        name (str): Logical name used in error messages.
        edge_index (torch.Tensor): Edge index tensor expected as shape `[2, E]`.
        errors (list[str]): Mutable list where validation errors are appended.
        expected_sender_range (tuple[int, int] | None): Optional valid half-open
            range `[min, max)` for sender indices (`edge_index[0]`).
        expected_receiver_range (tuple[int, int] | None): Optional valid
            half-open range `[min, max)` for receiver indices
            (`edge_index[1]`).

    Returns:
        int: Number of edges `E` inferred from the tensor, or `0` if invalid.
    """
    if not isinstance(edge_index, torch.Tensor):
        errors.append(f"{name}: expected torch.Tensor, got {type(edge_index)}")
        return 0

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        errors.append(
            f"{name}: expected shape [2, E], got {tuple(edge_index.shape)}"
        )
        return 0

    if not _is_integer_tensor(edge_index):
        errors.append(f"{name}: expected integer dtype, got {edge_index.dtype}")
        return edge_index.shape[1]

    if edge_index.shape[1] == 0:
        errors.append(f"{name}: contains zero edges")
        return 0

    if int(edge_index.min().item()) < 0:
        errors.append(f"{name}: contains negative node indices")

    if expected_sender_range is not None:
        send_min, send_max = expected_sender_range
        senders = edge_index[0]
        if (
            int(senders.min().item()) < send_min
            or int(senders.max().item()) >= send_max
        ):
            errors.append(
                f"{name}: sender indices out of expected range "
                f"[{send_min}, {send_max})"
            )

    if expected_receiver_range is not None:
        rec_min, rec_max = expected_receiver_range
        receivers = edge_index[1]
        if (
            int(receivers.min().item()) < rec_min
            or int(receivers.max().item()) >= rec_max
        ):
            errors.append(
                f"{name}: receiver indices out of expected range "
                f"[{rec_min}, {rec_max})"
            )

    return edge_index.shape[1]


def _check_edge_features(
    *,
    name: str,
    features: torch.Tensor,
    expected_num_edges: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Validate edge feature tensor shape, dtype, and basic geometric consistency.

    Args:
        name (str): Logical name used in error/warning messages.
        features (torch.Tensor): Edge features expected as shape `[E, 3]`.
        expected_num_edges (int): Expected number of rows (`E`) to match edge
            index count.
        errors (list[str]): Mutable list where validation errors are appended.
        warnings (list[str]): Mutable list where non-fatal issues are appended.

    Returns:
        None: Results are reported by mutating `errors` and `warnings`.
    """
    if not isinstance(features, torch.Tensor):
        errors.append(f"{name}: expected torch.Tensor, got {type(features)}")
        return

    if features.ndim != 2:
        errors.append(f"{name}: expected shape [E, 3], got {features.shape}")
        return

    if features.shape[0] != expected_num_edges:
        errors.append(
            f"{name}: number of rows ({features.shape[0]}) must match number "
            f"of edges ({expected_num_edges})"
        )

    if features.shape[1] != EDGE_FEATURE_DIM:
        errors.append(
            f"{name}: expected feature dim {EDGE_FEATURE_DIM}, got "
            f"{features.shape[1]}"
        )
        return

    if not torch.is_floating_point(features):
        errors.append(f"{name}: expected floating-point dtype")
        return

    if not torch.isfinite(features).all():
        errors.append(f"{name}: contains non-finite values")
        return

    if torch.any(features[:, 0] < 0):
        errors.append(f"{name}: first column (edge length) has negative values")

    if features.shape[0] == 0:
        return

    vec_norm = torch.linalg.vector_norm(features[:, 1:], dim=1)
    max_diff = float(torch.max(torch.abs(vec_norm - features[:, 0])).item())
    if max_diff > 5.0e-3:
        warnings.append(
            f"{name}: ||vdiff|| and stored edge length differ "
            f"(max abs diff={max_diff:.3e})"
        )


def _check_mesh_features(
    *,
    name: str,
    mesh_features: torch.Tensor,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Validate mesh node feature tensor.

    Args:
        name (str): Logical name used in error/warning messages.
        mesh_features (torch.Tensor): Mesh node features expected as shape
            `[N, 2]`.
        errors (list[str]): Mutable list where validation errors are appended.
        warnings (list[str]): Mutable list where non-fatal issues are appended.

    Returns:
        None: Results are reported by mutating `errors` and `warnings`.
    """
    if not isinstance(mesh_features, torch.Tensor):
        errors.append(
            f"{name}: expected torch.Tensor, got {type(mesh_features)}"
        )
        return

    if mesh_features.ndim != 2:
        errors.append(
            f"{name}: expected shape [N, 2], got {mesh_features.shape}"
        )
        return

    if mesh_features.shape[0] == 0:
        errors.append(f"{name}: contains zero mesh nodes")

    if mesh_features.shape[1] != MESH_FEATURE_DIM:
        errors.append(
            f"{name}: expected feature dim {MESH_FEATURE_DIM}, got "
            f"{mesh_features.shape[1]}"
        )
        return

    if not torch.is_floating_point(mesh_features):
        errors.append(f"{name}: expected floating-point dtype")
        return

    if not torch.isfinite(mesh_features).all():
        errors.append(f"{name}: contains non-finite values")
        return


def _require_file(path: Path, errors: list[str]) -> bool:
    """
    Assert that a required file exists.

    Args:
        path (Path): Path to required file.
        errors (list[str]): Mutable list where missing-file errors are appended.

    Returns:
        bool: `True` if file exists, `False` otherwise.
    """
    if not path.exists():
        errors.append(f"missing required file: {path.name}")
        return False
    return True


def validate_graph_directory(
    graph_dir_path: str | Path,
) -> GraphValidationReport:
    """
    Validate a neural-lam graph directory stored on disk.

    Args:
        graph_dir_path (str | Path): Path to graph directory containing required
            `.pt` components.

    Returns:
        GraphValidationReport: Structured validation result, including metadata,
        errors, and warnings.
    """
    graph_dir = Path(graph_dir_path)
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

    for file_name in required_files:
        _require_file(graph_dir / file_name, errors)
    if errors:
        return GraphValidationReport(
            graph_dir=str(graph_dir),
            hierarchical=False,
            num_levels=0,
            num_mesh_nodes_per_level=[],
            num_mesh_nodes_total=0,
            num_grid_nodes=0,
            errors=errors,
            warnings=warnings,
        )

    m2m_edge_index = _load_pt(graph_dir / "m2m_edge_index.pt")
    m2m_features = _load_pt(graph_dir / "m2m_features.pt")
    mesh_features = _load_pt(graph_dir / "mesh_features.pt")
    g2m_edge_index = _load_pt(graph_dir / "g2m_edge_index.pt")
    g2m_features = _load_pt(graph_dir / "g2m_features.pt")
    m2g_edge_index = _load_pt(graph_dir / "m2g_edge_index.pt")
    m2g_features = _load_pt(graph_dir / "m2g_features.pt")

    if not isinstance(m2m_edge_index, list):
        errors.append("m2m_edge_index.pt: expected list[Tensor]")
    if not isinstance(m2m_features, list):
        errors.append("m2m_features.pt: expected list[Tensor]")
    if not isinstance(mesh_features, list):
        errors.append("mesh_features.pt: expected list[Tensor]")
    if errors:
        return GraphValidationReport(
            graph_dir=str(graph_dir),
            hierarchical=False,
            num_levels=0,
            num_mesh_nodes_per_level=[],
            num_mesh_nodes_total=0,
            num_grid_nodes=0,
            errors=errors,
            warnings=warnings,
        )

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1
    if n_levels == 0:
        errors.append("m2m_edge_index.pt: list must not be empty")

    if len(m2m_features) != n_levels:
        errors.append(
            "m2m_features.pt: number of levels must match m2m_edge_index.pt"
        )
    if len(mesh_features) != n_levels:
        errors.append(
            "mesh_features.pt: number of levels must match m2m_edge_index.pt"
        )

    mesh_nodes_per_level: list[int] = []
    for level_index in range(min(n_levels, len(mesh_features))):
        mesh_tensor = mesh_features[level_index]
        _check_mesh_features(
            name=f"mesh_features[{level_index}]",
            mesh_features=mesh_tensor,
            errors=errors,
            warnings=warnings,
        )
        if isinstance(mesh_tensor, torch.Tensor) and mesh_tensor.ndim == 2:
            mesh_nodes_per_level.append(int(mesh_tensor.shape[0]))

    level_offsets: list[int] = []
    cumulative = 0
    for n_level_nodes in mesh_nodes_per_level:
        level_offsets.append(cumulative)
        cumulative += n_level_nodes
    num_mesh_nodes_total = cumulative

    for level_index in range(min(n_levels, len(m2m_edge_index))):
        level_edge_index = m2m_edge_index[level_index]
        if level_index < len(level_offsets):
            start = level_offsets[level_index]
            stop = start + mesh_nodes_per_level[level_index]
            expected_range = (start, stop)
        else:
            expected_range = None

        n_edges = _check_edge_index(
            name=f"m2m_edge_index[{level_index}]",
            edge_index=level_edge_index,
            errors=errors,
            expected_sender_range=expected_range,
            expected_receiver_range=expected_range,
        )

        if level_index < len(m2m_features):
            _check_edge_features(
                name=f"m2m_features[{level_index}]",
                features=m2m_features[level_index],
                expected_num_edges=n_edges,
                errors=errors,
                warnings=warnings,
            )

    extra_required_if_hier = [
        "mesh_up_edge_index.pt",
        "mesh_down_edge_index.pt",
        "mesh_up_features.pt",
        "mesh_down_features.pt",
    ]
    if hierarchical:
        for file_name in extra_required_if_hier:
            _require_file(graph_dir / file_name, errors)
        if not errors:
            mesh_up_edge_index = _load_pt(graph_dir / "mesh_up_edge_index.pt")
            mesh_down_edge_index = _load_pt(
                graph_dir / "mesh_down_edge_index.pt"
            )
            mesh_up_features = _load_pt(graph_dir / "mesh_up_features.pt")
            mesh_down_features = _load_pt(graph_dir / "mesh_down_features.pt")

            pairs = [
                ("mesh_up_edge_index.pt", mesh_up_edge_index),
                ("mesh_down_edge_index.pt", mesh_down_edge_index),
                ("mesh_up_features.pt", mesh_up_features),
                ("mesh_down_features.pt", mesh_down_features),
            ]
            for file_name, obj in pairs:
                if not isinstance(obj, list):
                    errors.append(f"{file_name}: expected list[Tensor]")

            expected_len = n_levels - 1
            if isinstance(mesh_up_edge_index, list) and (
                len(mesh_up_edge_index) != expected_len
            ):
                errors.append(
                    "mesh_up_edge_index.pt: expected length "
                    f"{expected_len}, got {len(mesh_up_edge_index)}"
                )
            if isinstance(mesh_down_edge_index, list) and (
                len(mesh_down_edge_index) != expected_len
            ):
                errors.append(
                    "mesh_down_edge_index.pt: expected length "
                    f"{expected_len}, got {len(mesh_down_edge_index)}"
                )
            if isinstance(mesh_up_features, list) and (
                len(mesh_up_features) != expected_len
            ):
                errors.append(
                    "mesh_up_features.pt: expected length "
                    f"{expected_len}, got {len(mesh_up_features)}"
                )
            if isinstance(mesh_down_features, list) and (
                len(mesh_down_features) != expected_len
            ):
                errors.append(
                    "mesh_down_features.pt: expected length "
                    f"{expected_len}, got {len(mesh_down_features)}"
                )

            if len(mesh_nodes_per_level) != n_levels:
                errors.append(
                    "cannot validate hierarchical up/down index ranges "
                    "because mesh_features is malformed"
                )
            for level_index in range(expected_len):
                if len(mesh_nodes_per_level) != n_levels:
                    break
                lower_start = level_offsets[level_index]
                lower_stop = lower_start + mesh_nodes_per_level[level_index]
                upper_start = level_offsets[level_index + 1]
                upper_stop = upper_start + mesh_nodes_per_level[level_index + 1]

                n_up = _check_edge_index(
                    name=f"mesh_up_edge_index[{level_index}]",
                    edge_index=mesh_up_edge_index[level_index],
                    errors=errors,
                    expected_sender_range=(lower_start, lower_stop),
                    expected_receiver_range=(upper_start, upper_stop),
                )
                _check_edge_features(
                    name=f"mesh_up_features[{level_index}]",
                    features=mesh_up_features[level_index],
                    expected_num_edges=n_up,
                    errors=errors,
                    warnings=warnings,
                )

                n_down = _check_edge_index(
                    name=f"mesh_down_edge_index[{level_index}]",
                    edge_index=mesh_down_edge_index[level_index],
                    errors=errors,
                    expected_sender_range=(upper_start, upper_stop),
                    expected_receiver_range=(lower_start, lower_stop),
                )
                _check_edge_features(
                    name=f"mesh_down_features[{level_index}]",
                    features=mesh_down_features[level_index],
                    expected_num_edges=n_down,
                    errors=errors,
                    warnings=warnings,
                )

    n_g2m_edges = _check_edge_index(
        name="g2m_edge_index",
        edge_index=g2m_edge_index,
        errors=errors,
    )
    _check_edge_features(
        name="g2m_features",
        features=g2m_features,
        expected_num_edges=n_g2m_edges,
        errors=errors,
        warnings=warnings,
    )

    n_m2g_edges = _check_edge_index(
        name="m2g_edge_index",
        edge_index=m2g_edge_index,
        errors=errors,
    )
    _check_edge_features(
        name="m2g_features",
        features=m2g_features,
        expected_num_edges=n_m2g_edges,
        errors=errors,
        warnings=warnings,
    )

    num_grid_nodes = 0
    if isinstance(g2m_edge_index, torch.Tensor) and isinstance(
        m2g_edge_index, torch.Tensor
    ):
        if num_mesh_nodes_total > 0:
            m2g_receiver_min = int(m2g_edge_index[1].min().item())
            if m2g_receiver_min != num_mesh_nodes_total:
                errors.append(
                    "m2g_edge_index: expected receiver indices to start at "
                    f"{num_mesh_nodes_total}, got {m2g_receiver_min}"
                )

            if int(g2m_edge_index[1].max().item()) >= mesh_nodes_per_level[0]:
                errors.append(
                    "g2m_edge_index: expected receivers on bottom mesh level"
                )
            if int(m2g_edge_index[0].max().item()) >= mesh_nodes_per_level[0]:
                errors.append(
                    "m2g_edge_index: expected senders on bottom mesh level"
                )

            num_grid_nodes = (
                int(m2g_edge_index[1].max().item()) - num_mesh_nodes_total + 1
            )
            g2m_sender_min = int(g2m_edge_index[0].min().item())
            g2m_sender_max = int(g2m_edge_index[0].max().item())
            g2m_lower = num_mesh_nodes_total
            g2m_upper = num_mesh_nodes_total + num_grid_nodes
            if g2m_sender_min < g2m_lower or g2m_sender_max >= g2m_upper:
                errors.append(
                    "g2m_edge_index: sender indices outside inferred grid "
                    f"range [{g2m_lower}, {g2m_upper})"
                )

    return GraphValidationReport(
        graph_dir=str(graph_dir),
        hierarchical=hierarchical,
        num_levels=n_levels,
        num_mesh_nodes_per_level=mesh_nodes_per_level,
        num_mesh_nodes_total=num_mesh_nodes_total,
        num_grid_nodes=num_grid_nodes,
        errors=errors,
        warnings=warnings,
    )


def cli(input_args=None):
    """
    Command-line entry point for graph validation.

    Args:
        input_args (list[str] | None): Optional argument list. If `None`,
            arguments are read from `sys.argv`.

    Returns:
        int: Process exit code (`0` if validation passes, `1` otherwise).
    """
    parser = ArgumentParser(
        description="Validate on-disk neural-lam graph components",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Path to a graph directory to validate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    args = parser.parse_args(input_args)

    report = validate_graph_directory(args.graph_dir)

    if args.json:
        payload = asdict(report)
        payload["ok"] = report.ok
        print(json.dumps(payload, indent=2))
    else:
        print(f"Graph directory: {report.graph_dir}")
        print(f"Hierarchical: {report.hierarchical}")
        print(f"Mesh levels: {report.num_levels}")
        print(f"Mesh nodes per level: {report.num_mesh_nodes_per_level}")
        print(f"Mesh nodes total: {report.num_mesh_nodes_total}")
        print(f"Grid nodes (inferred): {report.num_grid_nodes}")
        if report.errors:
            print("Status: FAIL")
            for error in report.errors:
                print(f"ERROR: {error}")
        else:
            print("Status: PASS")
        for warning in report.warnings:
            print(f"WARNING: {warning}")

    if report.ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
