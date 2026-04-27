"""
Standalone CLI validator and specification for neural-lam on-disk graph directories.  # noqa: E501

Run with:
    uv run docs/validate_graph_new.py <path-to-graph-dir>
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24.2",
#   "torch>=2.3.0",
#   "rich>=13.0.0",
# ]
# ///

# Standard library
import json
import sys
import textwrap
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party
import torch

try:
    # Third-party
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


ALLOWED_EDGE_FEATURE_DIMS = (3, 4)
MESH_FEATURE_DIM = 2

# -------------------------
# Logging decorator and registry
# -------------------------
CHECK_REGISTRY: Dict[Tuple[str, str], Callable] = {}


def log_function_call(func):
    """
    Decorator to register function calls for spec rendering.

    The original callable is stored in `CHECK_REGISTRY` so that it can be
    monkey patched (e.g., when printing specs without running validations).

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function that registers its module and name on results.
    """
    if (func.__module__, func.__name__) not in CHECK_REGISTRY:
        CHECK_REGISTRY[(func.__module__, func.__name__)] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_from_registry = CHECK_REGISTRY[(func.__module__, func.__name__)]
        report = func_from_registry(*args, **kwargs)
        for result in report.results:
            result.module = func.__module__
            result.function = func.__name__
        return report

    return wrapper


# -------------------------
# Data structures
# -------------------------
@dataclass
class Result:
    """
    Structured result of a single validation check.

    Attributes
    ----------
    section : str
        The section where the result occurred.
    requirement : str
        The specific requirement that was evaluated.
    status : str
        "FAIL", "WARNING", or "PASS".
    detail : str
        Additional details about the result.
    module : str, optional
        The module where the checking function is located.
    function : str, optional
        The name of the checking function.
    """

    section: str
    requirement: str
    status: str
    detail: str = ""
    module: Optional[str] = None
    function: Optional[str] = None

    def __post_init__(self):
        valid_levels = {"FAIL", "WARNING", "PASS"}
        if self.status not in valid_levels:
            raise ValueError(
                f"Invalid status: {self.status}. Valid levels are: "
                f"{', '.join(valid_levels)}."
            )


@dataclass
class ValidationReport:
    """
    Container for accumulating validation results.

    Attributes
    ----------
    ok : bool
        Whether the validation passes (no fatal errors).
    results : list[Result]
        The collected validation results.
    """

    ok: bool = True
    results: List[Result] = field(default_factory=list)

    def add(
        self, section: str, requirement: str, status: str, detail: str = ""
    ) -> None:
        """
        Add a result to the validation report.

        Parameters
        ----------
        section : str
            The section where the result occurred.
        requirement : str
            The specific requirement that was evaluated.
        status : str
            The severity status of the result ("FAIL", "WARNING", "PASS").
        detail : str, optional
            Additional details about the result.

        Returns
        -------
        None
        """
        self.results.append(Result(section, requirement, status, detail))

    def summarize(self) -> str:
        """
        Summarize the validation report by counting results of each severity level.  # noqa: E501

        Returns
        -------
        str
            A summary string with counts of fails, warnings, and passes.
        """
        fails = sum(1 for r in self.results if r.status == "FAIL")
        warns = sum(1 for r in self.results if r.status == "WARNING")
        passes = sum(1 for r in self.results if r.status == "PASS")
        return (
            f"Summary: {fails} fail(s), {warns} warning(s), {passes} pass(es)."
        )

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        """
        Merge another ValidationReport into this one (in-place).

        Parameters
        ----------
        other : ValidationReport
            The other validation report to merge.

        Returns
        -------
        ValidationReport
            The updated validation report (self).
        """
        self.results.extend(other.results)
        self.ok = self.ok and other.ok
        return self

    def __add__(self, other: "ValidationReport") -> "ValidationReport":
        """
        Combine two ValidationReports into a new one.

        Parameters
        ----------
        other : ValidationReport
            The other validation report to combine.

        Returns
        -------
        ValidationReport
            A new validation report containing results from both.
        """
        out = ValidationReport(ok=self.ok and other.ok)
        out.results = [*self.results, *other.results]
        return out

    def console_print(self, *, file=None) -> None:
        """
        Print all results in the validation report to the console.
        Uses rich table if available, otherwise falls back to plain text.

        Parameters
        ----------
        file : file-like, optional
            Optional file-like object to write to (defaults to stdout).

        Returns
        -------
        None
        """
        if RICH_AVAILABLE:
            console = Console(file=file)
            table = Table(title="Validation Report")
            table.add_column("Section", style="bold")
            table.add_column("Requirement", style="dim")
            table.add_column("Status", justify="center")
            table.add_column("Detail", style="italic")
            table.add_column("Checking function", style="bold")

            level_emojis = {"FAIL": "❌", "WARNING": "⚠️", "PASS": "✅"}

            for result in self.results:
                fn_fqn = result.function if result.function else "N/A"
                table.add_row(
                    result.section,
                    result.requirement,
                    level_emojis.get(result.status, result.status),
                    result.detail,
                    fn_fqn,
                )

            console.print(table)
            console.print(self.summarize())
        else:
            out = file if file else sys.stdout
            print("Validation Report", file=out)
            print("-" * 80, file=out)
            for result in self.results:
                print(
                    f"[{result.status}] {result.section} - {result.requirement}",  # noqa: E501
                    file=out,
                )
                if result.detail:
                    print(f"    Detail: {result.detail}", file=out)
            print("-" * 80, file=out)
            print(self.summarize(), file=out)

    def has_fails(self) -> bool:
        """
        Check if the report contains any FAIL results.

        Returns
        -------
        bool
            True if there is at least one FAIL result, False otherwise.
        """
        return any(r.status == "FAIL" for r in self.results)

    def has_warnings(self) -> bool:
        """
        Check if the report contains any WARNING results.

        Returns
        -------
        bool
            True if there is at least one WARNING result, False otherwise.
        """
        return any(r.status == "WARNING" for r in self.results)


@dataclass
class GraphProperties:
    """
    Accumulator for inferred properties of the graph directory.

    Attributes
    ----------
    hierarchical : bool
        Whether the graph has hierarchical mesh levels.
    num_levels : int
        Number of mesh levels encoded in `m2m_*`/`mesh_*`.
    num_mesh_nodes_per_level : list[int]
        Mesh node count for each level.
    num_mesh_nodes_total : int
        Total number of mesh nodes across levels.
    num_grid_nodes : int
        Inferred number of grid nodes.
    """

    hierarchical: bool = False
    num_levels: int = 0
    num_mesh_nodes_per_level: list[int] = field(default_factory=list)
    num_mesh_nodes_total: int = 0
    num_grid_nodes: int = 0

    def __add__(self, other: "GraphProperties") -> "GraphProperties":
        """
        Combine two GraphProperties objects. Returns a new object taking the
        non-default properties from either side (assumes they don't conflict).

        Parameters
        ----------
        other : GraphProperties
            The other properties to merge.

        Returns
        -------
        GraphProperties
            A new merged properties object.
        """
        return GraphProperties(
            hierarchical=self.hierarchical or other.hierarchical,
            num_levels=max(self.num_levels, other.num_levels),
            num_mesh_nodes_per_level=self.num_mesh_nodes_per_level
            or other.num_mesh_nodes_per_level,
            num_mesh_nodes_total=max(
                self.num_mesh_nodes_total, other.num_mesh_nodes_total
            ),
            num_grid_nodes=max(self.num_grid_nodes, other.num_grid_nodes),
        )

    def __iadd__(self, other: "GraphProperties") -> "GraphProperties":
        self.hierarchical = self.hierarchical or other.hierarchical
        self.num_levels = max(self.num_levels, other.num_levels)
        if other.num_mesh_nodes_per_level:
            self.num_mesh_nodes_per_level = other.num_mesh_nodes_per_level
        self.num_mesh_nodes_total = max(
            self.num_mesh_nodes_total, other.num_mesh_nodes_total
        )
        self.num_grid_nodes = max(self.num_grid_nodes, other.num_grid_nodes)
        return self


@contextmanager
def skip_all_checks():
    """
    Context manager to bypass check functions and tensor loading.

    Assumes check functions are decorated with `log_function_call` which
    records them in `CHECK_REGISTRY`. Each registered check is monkey patched
    to a stub returning an empty `ValidationReport`. `_load_pt` is also stubbed
    to avoid reading from disk.
    """

    def _stubbed_check(*_args, **_kwargs):
        return ValidationReport()

    def _stubbed_load_pt(*_args, **_kwargs):
        return None

    with ExitStack() as stack:
        # Standard library
        from unittest import mock

        stack.enter_context(
            mock.patch(f"{__name__}._load_pt", _stubbed_load_pt)
        )

        stack.enter_context(
            mock.patch.dict(
                CHECK_REGISTRY,
                {key: _stubbed_check for key in list(CHECK_REGISTRY.keys())},
            )
        )
        yield


# -------------------------
# Core Loaders and Inference
# -------------------------
def _load_pt(path: Path) -> Any:
    """
    Load a torch-serialized object from disk.

    Parameters
    ----------
    path : Path
        Path to a `.pt` file.

    Returns
    -------
    Any
        Deserialized object from the `.pt` file.
    """
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def infer_levels(m2m_edge_index: Any) -> GraphProperties:
    """
    Infer the number of mesh levels from the `m2m_edge_index` list.

    Parameters
    ----------
    m2m_edge_index : Any
        The loaded m2m edge indices.

    Returns
    -------
    GraphProperties
        Properties object with inferred `num_levels` and `hierarchical`.
    """
    props = GraphProperties()
    if isinstance(m2m_edge_index, list):
        props.num_levels = len(m2m_edge_index)
        props.hierarchical = props.num_levels > 1
    return props


def infer_mesh_nodes_per_level(mesh_features: Any) -> GraphProperties:
    """
    Infer the number of mesh nodes per level from `mesh_features`.

    Parameters
    ----------
    mesh_features : Any
        The loaded mesh features.

    Returns
    -------
    GraphProperties
        Properties object with inferred `num_mesh_nodes_per_level` and
        `num_mesh_nodes_total`.
    """
    props = GraphProperties()
    if isinstance(mesh_features, list):
        nodes_per_level = []
        for tensor in mesh_features:
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                nodes_per_level.append(int(tensor.shape[0]))
        props.num_mesh_nodes_per_level = nodes_per_level
        props.num_mesh_nodes_total = sum(nodes_per_level)
    return props


def infer_grid_nodes(
    m2g_edge_index: Any, num_mesh_nodes_total: int
) -> GraphProperties:
    """
    Infer the total number of grid nodes based on the receiver indices in `m2g_edge_index`.  # noqa: E501

    Parameters
    ----------
    m2g_edge_index : Any
        The loaded m2g edge indices.
    num_mesh_nodes_total : int
        The previously inferred total number of mesh nodes.

    Returns
    -------
    GraphProperties
        Properties object with inferred `num_grid_nodes`.
    """
    props = GraphProperties()
    if isinstance(m2g_edge_index, torch.Tensor) and num_mesh_nodes_total > 0:
        if (
            m2g_edge_index.ndim == 2
            and m2g_edge_index.shape[0] == 2
            and m2g_edge_index.shape[1] > 0
        ):
            props.num_grid_nodes = (
                int(m2g_edge_index[1].max().item()) - num_mesh_nodes_total + 1
            )
    return props


# -------------------------
# Check Functions
# -------------------------
@log_function_call
def check_required_files(
    graph_dir: Path, files: List[str], section_name: str
) -> ValidationReport:
    """
    Verify that all files in a provided list exist in the graph directory.

    Parameters
    ----------
    graph_dir : Path
        Directory containing the graph.
    files : list of str
        List of filenames that must exist.
    section_name : str
        The spec section name for reporting.

    Returns
    -------
    ValidationReport
        The report indicating if the files are present.
    """
    report = ValidationReport()
    missing = [f for f in files if not (graph_dir / f).exists()]
    if missing:
        report.add(
            section_name,
            "Required files presence",
            "FAIL",
            f"Missing required files: {', '.join(missing)}",
        )
    else:
        report.add(
            section_name,
            "Required files presence",
            "PASS",
            "All required files are present",
        )
    return report


@log_function_call
def check_list_type_and_length(
    obj: Any,
    name: str,
    expected_length: int,
    section_name: str,
    allow_empty: bool = False,
) -> ValidationReport:
    """
    Verify that an object is a list of the expected length.

    Parameters
    ----------
    obj : Any
        The object to check.
    name : str
        Logical name of the object.
    expected_length : int
        The length the list should be.
    section_name : str
        The spec section name for reporting.
    allow_empty : bool, optional
        Whether an empty list is allowed regardless of expected length.

    Returns
    -------
    ValidationReport
        The report indicating if the list properties are valid.
    """
    report = ValidationReport()
    if obj is None:
        return report  # If missing, covered by file existence checks

    if not isinstance(obj, list):
        report.add(
            section_name,
            "List format",
            "FAIL",
            f"{name}: expected list[Tensor], got {type(obj)}",
        )
        return report

    if not allow_empty and len(obj) == 0:
        report.add(
            section_name,
            "List format",
            "FAIL",
            f"{name}: list must not be empty",
        )
        return report

    if len(obj) != expected_length:
        report.add(
            section_name,
            "List length",
            "FAIL",
            f"{name}: expected length {expected_length}, got {len(obj)}",
        )
    else:
        report.add(
            section_name,
            "List length",
            "PASS",
            f"{name}: length matches expected ({expected_length})",
        )

    return report


@log_function_call
def check_edge_index(
    name: str,
    edge_index: Any,
    section_name: str,
    expected_sender_range: tuple[int, int] | None = None,
    expected_receiver_range: tuple[int, int] | None = None,
) -> ValidationReport:
    """
    Validate edge index tensor shape, dtype, and optional index ranges.

    Parameters
    ----------
    name : str
        Logical name used in error messages.
    edge_index : Any
        Edge index tensor expected as shape `[2, E]`.
    section_name : str
        The spec section name for reporting.
    expected_sender_range : tuple[int, int] | None, optional
        Optional valid half-open range `[min, max)` for sender indices.
    expected_receiver_range : tuple[int, int] | None, optional
        Optional valid half-open range `[min, max)` for receiver indices.

    Returns
    -------
    ValidationReport
        The validation report.
    """
    report = ValidationReport()
    if edge_index is None:
        return report

    if not isinstance(edge_index, torch.Tensor):
        report.add(
            section_name,
            "Edge index format",
            "FAIL",
            f"{name}: expected torch.Tensor, got {type(edge_index)}",
        )
        return report

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        report.add(
            section_name,
            "Edge index shape",
            "FAIL",
            f"{name}: expected shape [2, E], got {tuple(edge_index.shape)}",
        )
        return report

    report.add(
        section_name,
        "Edge index shape",
        "PASS",
        f"{name}: shape is {tuple(edge_index.shape)}",
    )

    # Check for integer dtype
    if edge_index.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        report.add(
            section_name,
            "Edge index dtype",
            "FAIL",
            f"{name}: expected integer dtype, got {edge_index.dtype}",
        )
    else:
        report.add(
            section_name,
            "Edge index dtype",
            "PASS",
            f"{name}: dtype is integer",
        )

    if edge_index.shape[1] == 0:
        report.add(
            section_name,
            "Edge index values",
            "FAIL",
            f"{name}: contains zero edges",
        )
        return report

    if int(edge_index.min().item()) < 0:
        report.add(
            section_name,
            "Edge index values",
            "FAIL",
            f"{name}: contains negative node indices",
        )

    if expected_sender_range is not None:
        send_min, send_max = expected_sender_range
        senders = edge_index[0]
        if (
            int(senders.min().item()) < send_min
            or int(senders.max().item()) >= send_max
        ):
            report.add(
                section_name,
                "Sender index range",
                "FAIL",
                f"{name}: sender indices out of expected range [{send_min}, {send_max})",  # noqa: E501
            )

    if expected_receiver_range is not None:
        rec_min, rec_max = expected_receiver_range
        receivers = edge_index[1]
        if (
            int(receivers.min().item()) < rec_min
            or int(receivers.max().item()) >= rec_max
        ):
            report.add(
                section_name,
                "Receiver index range",
                "FAIL",
                f"{name}: receiver indices out of expected range [{rec_min}, {rec_max})",  # noqa: E501
            )

    return report


@log_function_call
def check_edge_features(
    name: str,
    features: Any,
    expected_num_edges: int | None,
    section_name: str,
) -> ValidationReport:
    """
    Validate edge feature tensor shape, dtype, and basic geometric consistency.

    Parameters
    ----------
    name : str
        Logical name used in error/warning messages.
    features : Any
        Edge features expected as shape `[E, 3 or 4]`.
    expected_num_edges : int | None
        Expected number of rows (`E`) to match edge index count.
    section_name : str
        The spec section name for reporting.

    Returns
    -------
    ValidationReport
        The validation report.
    """
    report = ValidationReport()
    if features is None:
        return report

    if not isinstance(features, torch.Tensor):
        report.add(
            section_name,
            "Edge features format",
            "FAIL",
            f"{name}: expected torch.Tensor, got {type(features)}",
        )
        return report

    if features.ndim != 2:
        report.add(
            section_name,
            "Edge features shape",
            "FAIL",
            f"{name}: expected shape [E, 3 or 4], got {tuple(features.shape)}",
        )
        return report

    if (
        expected_num_edges is not None
        and features.shape[0] != expected_num_edges
    ):
        report.add(
            section_name,
            "Edge features shape",
            "FAIL",
            f"{name}: number of rows ({features.shape[0]}) must match number of edges ({expected_num_edges})",  # noqa: E501
        )
    else:
        report.add(
            section_name,
            "Edge features shape",
            "PASS",
            f"{name}: number of rows matches edges",
        )

    if features.shape[1] not in ALLOWED_EDGE_FEATURE_DIMS:
        report.add(
            section_name,
            "Edge features shape",
            "FAIL",
            f"{name}: expected feature dim in {ALLOWED_EDGE_FEATURE_DIMS}, got {features.shape[1]}",  # noqa: E501
        )
    else:
        report.add(
            section_name,
            "Edge features shape",
            "PASS",
            f"{name}: feature dim is {features.shape[1]}",
        )

    if features.dtype != torch.float32:
        report.add(
            section_name,
            "Edge features dtype",
            "FAIL",
            f"{name}: expected torch.float32 dtype, got {features.dtype}",
        )
    else:
        report.add(
            section_name,
            "Edge features dtype",
            "PASS",
            f"{name}: dtype is torch.float32",
        )

    if not torch.isfinite(features).all():
        report.add(
            section_name,
            "Edge features values",
            "FAIL",
            f"{name}: contains non-finite values",
        )
        return report

    if torch.any(features[:, 0] < 0):
        report.add(
            section_name,
            "Edge features values",
            "FAIL",
            f"{name}: first column (edge length) has negative values",
        )

    if features.shape[0] > 0:
        vec_norm = torch.linalg.vector_norm(features[:, 1:], dim=1)
        max_diff = float(torch.max(torch.abs(vec_norm - features[:, 0])).item())
        if max_diff > 5.0e-3:
            report.add(
                section_name,
                "Edge features geometry",
                "WARNING",
                f"{name}: ||vdiff|| and stored edge length differ (max abs diff={max_diff:.3e})",  # noqa: E501
            )

    return report


@log_function_call
def check_mesh_features(
    name: str,
    mesh_features: Any,
    section_name: str,
) -> ValidationReport:
    """
    Validate mesh node feature tensor shape and dtype.

    Parameters
    ----------
    name : str
        Logical name used in error/warning messages.
    mesh_features : Any
        Mesh node features expected as shape `[N, >=2]`.
    section_name : str
        The spec section name for reporting.

    Returns
    -------
    ValidationReport
        The validation report.
    """
    report = ValidationReport()
    if mesh_features is None:
        return report

    if not isinstance(mesh_features, torch.Tensor):
        report.add(
            section_name,
            "Mesh features format",
            "FAIL",
            f"{name}: expected torch.Tensor, got {type(mesh_features)}",
        )
        return report

    if mesh_features.ndim != 2:
        report.add(
            section_name,
            "Mesh features shape",
            "FAIL",
            f"{name}: expected shape [N, >=2], got {tuple(mesh_features.shape)}",  # noqa: E501
        )
        return report

    if mesh_features.shape[0] == 0:
        report.add(
            section_name,
            "Mesh features values",
            "FAIL",
            f"{name}: contains zero mesh nodes",
        )

    if mesh_features.shape[1] < MESH_FEATURE_DIM:
        report.add(
            section_name,
            "Mesh features shape",
            "FAIL",
            f"{name}: expected at least {MESH_FEATURE_DIM} features, got {mesh_features.shape[1]}",  # noqa: E501
        )
    else:
        report.add(
            section_name,
            "Mesh features shape",
            "PASS",
            f"{name}: features dimensionality is {mesh_features.shape[1]}",
        )

    if mesh_features.dtype != torch.float32:
        report.add(
            section_name,
            "Mesh features dtype",
            "FAIL",
            f"{name}: expected torch.float32 dtype, got {mesh_features.dtype}",
        )
    else:
        report.add(
            section_name,
            "Mesh features dtype",
            "PASS",
            f"{name}: dtype is torch.float32",
        )

    if not torch.isfinite(mesh_features).all():
        report.add(
            section_name,
            "Mesh features values",
            "FAIL",
            f"{name}: contains non-finite values",
        )

    return report


@log_function_call
def check_edge_feature_dim_consistency(
    named_features: list[tuple[str, Any]],
    section_name: str,
) -> ValidationReport:
    """
    Ensure all edge feature tensors have the same dimensionality.

    Parameters
    ----------
    named_features : list[tuple[str, Any]]
        List of (name, feature_tensor) tuples.
    section_name : str
        The spec section name for reporting.

    Returns
    -------
    ValidationReport
        The validation report.
    """
    report = ValidationReport()
    dims = {}
    for name, feats in named_features:
        if isinstance(feats, torch.Tensor) and feats.ndim == 2:
            dims[name] = feats.shape[1]

    unique_dims = set(dims.values())
    if len(unique_dims) > 1:
        summary = ", ".join(f"{n}={d}" for n, d in dims.items())
        report.add(
            section_name,
            "Edge feature dimensionality consistency",
            "FAIL",
            f"Mismatch across components: {summary}",
        )
    elif len(unique_dims) == 1:
        report.add(
            section_name,
            "Edge feature dimensionality consistency",
            "PASS",
            f"All components share N_f={list(unique_dims)[0]}",
        )

    return report


@log_function_call
def check_grid_node_relationships(
    g2m_edge_index: Any,
    m2g_edge_index: Any,
    mesh_nodes_per_level: list[int],
    num_mesh_nodes_total: int,
    num_grid_nodes: int,
    section_name: str,
) -> ValidationReport:
    """
    Check the inferred grid node ranges against indices in g2m and m2g.

    Parameters
    ----------
    g2m_edge_index : Any
        g2m edge indices tensor.
    m2g_edge_index : Any
        m2g edge indices tensor.
    mesh_nodes_per_level : list[int]
        Number of nodes at each mesh level.
    num_mesh_nodes_total : int
        Total number of mesh nodes.
    num_grid_nodes : int
        Total number of grid nodes inferred.
    section_name : str
        The spec section name for reporting.

    Returns
    -------
    ValidationReport
        The validation report.
    """
    report = ValidationReport()

    if not (
        isinstance(g2m_edge_index, torch.Tensor)
        and isinstance(m2g_edge_index, torch.Tensor)
    ):
        return report

    if num_mesh_nodes_total == 0 or len(mesh_nodes_per_level) == 0:
        return report

    if (
        g2m_edge_index.ndim != 2
        or g2m_edge_index.shape[0] != 2
        or g2m_edge_index.shape[1] == 0
    ):
        return report

    if (
        m2g_edge_index.ndim != 2
        or m2g_edge_index.shape[0] != 2
        or m2g_edge_index.shape[1] == 0
    ):
        return report

    m2g_receiver_min = int(m2g_edge_index[1].min().item())
    if m2g_receiver_min != num_mesh_nodes_total:
        report.add(
            section_name,
            "Grid node relationships",
            "FAIL",
            f"m2g_edge_index: expected receiver indices to start at {num_mesh_nodes_total}, got {m2g_receiver_min}",  # noqa: E501
        )

    if int(g2m_edge_index[1].max().item()) >= mesh_nodes_per_level[0]:
        report.add(
            section_name,
            "Grid node relationships",
            "FAIL",
            "g2m_edge_index: expected receivers on bottom mesh level",
        )

    if int(m2g_edge_index[0].max().item()) >= mesh_nodes_per_level[0]:
        report.add(
            section_name,
            "Grid node relationships",
            "FAIL",
            "m2g_edge_index: expected senders on bottom mesh level",
        )

    g2m_sender_min = int(g2m_edge_index[0].min().item())
    g2m_sender_max = int(g2m_edge_index[0].max().item())
    g2m_lower = num_mesh_nodes_total
    g2m_upper = num_mesh_nodes_total + num_grid_nodes
    if g2m_sender_min < g2m_lower or g2m_sender_max >= g2m_upper:
        report.add(
            section_name,
            "Grid node relationships",
            "FAIL",
            f"g2m_edge_index: sender indices outside inferred grid range [{g2m_lower}, {g2m_upper})",  # noqa: E501
        )

    return report


# -------------------------
# Spec Orchestrator
# -------------------------
def validate_graph_directory(
    graph_dir_path: str | Path | None,
) -> Tuple[ValidationReport, str, GraphProperties]:
    """
    Validate a neural-lam graph directory against the specification.
    Also produces the markdown specification text.

    Parameters
    ----------
    graph_dir_path : str | Path | None
        Path to graph directory. If None, validation is skipped and only the
        markdown spec is returned (useful with `skip_all_checks`).

    Returns
    -------
    ValidationReport
        The accumulated validation report.
    str
        The markdown specification text.
    GraphProperties
        The accumulated inferred properties of the graph.
    """
    report = ValidationReport()
    props = GraphProperties()
    graph_dir = Path(graph_dir_path) if graph_dir_path else None
    edge_feature_tensors = []

    spec_text = textwrap.dedent(
        """\
    # Neural-LAM Graph Storage Specification

    Version: 0.1.0-draft

    ## 1. Introduction

    This document specifies the requirements for Graph disk format for `neural-lam`.  # noqa: E501
    These graphs are used by the Neural-LAM Graph Neural Network architectures for  # noqa: E501
    machine-learning weather prediction (MLWP) forecasting. These model
    architectures follow the encode-process-decode paradigm of sequential message  # noqa: E501
    passing, where physical variables are represented as features on so-called
    *grid* nodes, are *encoded* to *mesh* nodes, are *processed* on the mesh, and  # noqa: E501
    then *decoded* back to grid nodes where output tendencies or updated state are  # noqa: E501
    produced.

    The format specified in this document was designed to support the definition of  # noqa: E501
    both flat (e.g. Keisler 2022, Lam et al 2022) and hierarchical (Oskarsson et al  # noqa: E501
    2023) graphs for GNN-based MLWP in neural-lam.

    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
    "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
    document are to be interpreted as described in RFC 2119.

    ## 2. File and Directory Structure

    ### Directory Structure

    Each graph MUST identified by a unique `name` and stored within the directory  # noqa: E501
    `graph/<name>/` that in turn MUST be placed within the same directory as the
    datastore configuration from which the graph was derived (i.e. the spatial
    coordinates defining the `grid` coordinates are provided by the datastore).
    """
    )

    spec_text += textwrap.dedent(
        """\
    ### Graph Filenames

    Each graph MUST be represented by multiple files in its graph directory.
    Together, these files define connectivity and static features for edges and
    mesh nodes.

    Required files for all graphs (all of these files MUST be present):

    - `m2m_edge_index.pt`
    - `g2m_edge_index.pt`
    - `m2g_edge_index.pt`
    - `m2m_features.pt`
    - `g2m_features.pt`
    - `m2g_features.pt`
    - `mesh_features.pt`
    """
    )

    required_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]

    if graph_dir and not graph_dir.exists():
        report.add(
            "2. File and Directory Structure",
            "Directory exists",
            "FAIL",
            f"Graph directory does not exist: {graph_dir}",
        )
        return report, spec_text, props
    elif graph_dir:
        report += check_required_files(
            graph_dir, required_files, "2. File and Directory Structure"
        )

    m2m_edge_index = (
        _load_pt(graph_dir / "m2m_edge_index.pt") if graph_dir else None
    )
    m2m_features = (
        _load_pt(graph_dir / "m2m_features.pt") if graph_dir else None
    )
    mesh_features = (
        _load_pt(graph_dir / "mesh_features.pt") if graph_dir else None
    )
    g2m_edge_index = (
        _load_pt(graph_dir / "g2m_edge_index.pt") if graph_dir else None
    )
    g2m_features = (
        _load_pt(graph_dir / "g2m_features.pt") if graph_dir else None
    )
    m2g_edge_index = (
        _load_pt(graph_dir / "m2g_edge_index.pt") if graph_dir else None
    )
    m2g_features = (
        _load_pt(graph_dir / "m2g_features.pt") if graph_dir else None
    )

    # Inference steps
    props += infer_levels(m2m_edge_index)
    props += infer_mesh_nodes_per_level(mesh_features)
    props += infer_grid_nodes(m2g_edge_index, props.num_mesh_nodes_total)

    spec_text += textwrap.dedent(
        """\
    Additional required files for hierarchical graphs (`L > 1` mesh levels), all
    of which MUST be present:

    - `mesh_up_edge_index.pt`
    - `mesh_down_edge_index.pt`
    - `mesh_up_features.pt`
    - `mesh_down_features.pt`
    """
    )

    extra_required_if_hier = [
        "mesh_up_edge_index.pt",
        "mesh_down_edge_index.pt",
        "mesh_up_features.pt",
        "mesh_down_features.pt",
    ]
    if graph_dir and props.hierarchical:
        report += check_required_files(
            graph_dir, extra_required_if_hier, "2. File and Directory Structure"
        )

    mesh_up_edge_index = (
        _load_pt(graph_dir / "mesh_up_edge_index.pt")
        if graph_dir and props.hierarchical
        else None
    )
    mesh_down_edge_index = (
        _load_pt(graph_dir / "mesh_down_edge_index.pt")
        if graph_dir and props.hierarchical
        else None
    )
    mesh_up_features = (
        _load_pt(graph_dir / "mesh_up_features.pt")
        if graph_dir and props.hierarchical
        else None
    )
    mesh_down_features = (
        _load_pt(graph_dir / "mesh_down_features.pt")
        if graph_dir and props.hierarchical
        else None
    )

    spec_text += textwrap.dedent(
        """\
    The separate files represent the different "components" of the graph, where
    each of the sequential message passing steps, `encode`, `process`, and
    `decode` uses a separate component, so that `g2m` is used in `encode`, `m2m`
    is used in `process`, and `m2g` is used in `decode`. For hierarchical graphs,  # noqa: E501
    the `m2m` component is further split into separate inter-level and intra-level  # noqa: E501
    message-passing steps.

    Each graph component MUST be represented by two files: one for edge
    connectivity and one for edge features. The components are (which also define  # noqa: E501
    the expected file prefixes):

    - `g2m`: grid-to-mesh edges (sender on grid, receiver on mesh).
    - `m2m`: mesh-to-mesh edges (both sender and receiver on mesh).
    - `m2g`: mesh-to-grid edges (sender on mesh, receiver on grid).
    - `mesh_up`: inter-level mesh edges from lower level to upper level.
    - `mesh_down`: inter-level mesh edges from upper level to lower level.

    Suffixes indicate content type:

    - `_edge_index.pt`: edge connectivity
    - `_features.pt`: static features associated with each edge
    - `mesh_features.pt`: static mesh node features

    All files MUST be serialized with `torch.save(...)`.

    > **NOTE**: Rather than the inter-level tensors files being prefixed with
    > `m2m`, they are prefixed with `mesh`, even though they are part of the
    > mesh-to-mesh message passing.

    ## 3. File content requirements

    The content of the files depend on the number of mesh levels, denoted as `L` in  # noqa: E501
    the text below, so that for:

    - Non-hierarchical graphs `L == 1`.
    - Hierarchical graphs `L > 1`.
    - Entry `0` MUST always be the bottom mesh level.

    Every tensors MUST stored in a manner ameanable to load with `torch.load(...)` (this can most easily be support by using `torch.save(...)` to store tensors to disk) and satisfy the requirements below.  # noqa: E501

    ### Nodes

    The `neural-lam` graph format on disk does not explicitly store node features for grid nodes, as these are expected to be dynamic and stored separately in the dataset. However, static features for mesh nodes MUST be stored in `mesh_features.pt` files (as described below).  # noqa: E501

    #### Node index space

    The node indices in edge index tensors MUST be defined so that for each nodeset (for example "mesh nodes level 0") the indices run from `0` to `N-1`, where `N` is the number of nodes in that nodeset, i.e. the node indices for each nodeset MUST be contiguous. For example, if there are `N_0` mesh nodes at level `0`, then the node indices for those nodes MUST be `0` to `N_0 - 1`. If there are `N_1` mesh nodes at level `1`, then the node indices for those nodes MUST be `N_0` to `N_0 + N_1 - 1`, and so on.  # noqa: E501

    NOTE: There is no requirement that the node indices for different nodesets be non-overlapping, in fact they should be overlapping, as the node indices for each nodeset are defined to run from `0` to `N-1` for that nodeset. The key requirement is that the node indices for each nodeset are contiguous and defined in a consistent manner across all edge index tensors.  # noqa: E501

    #### Mesh node features

    `mesh_features.pt` files MUST be lists of length `L` (number of mesh levels),  # noqa: E501
    where each entry is a tensor containing static features for the mesh nodes at  # noqa: E501
    that level. Each tensor MUST satisfy the following requirements:

    - `mesh_features` entries MUST have shape `[N_level, N_f]`, where `N_level` is the number of mesh nodes at that level and `N_f` is the number of features per node. `N_f` MUST be at minimum `2` (for x and y coordinates of the node, see next point), but can be larger if additional static features are included. The value of `N_f` MUST be consistent across all levels, so that all entries in the list have the same number of features per node.  # noqa: E501
    - `mesh_features[i][:, 0:2]` MUST contain the x and y coordinates of the mesh nodes at level `i`, so that column `0` is x and column `1` is y.  # noqa: E501
    - Mesh node features SHOULD NOT be normalized. Instead, normalization will be performed inside `neural-lam` after graph loading.  # noqa: E501
    - Dtype MUST be `torch.float32`.

    *NOTE*: The reason for requiring that the first two columns of the mesh node features contain the x and y coordinates is that `neural-lam` applies different normalization strategies to coordinates vs. extra features. The first two columns (coordinates) share the same spatial scale and are normalized jointly by their maximum absolute value across all levels. Any additional feature columns (index 2 onwards) are normalized independently by their own maximum absolute values.  # noqa: E501
    """
    )

    report += check_list_type_and_length(
        mesh_features,
        "mesh_features.pt",
        props.num_levels,
        "3. File content requirements (Mesh Nodes)",
    )

    if isinstance(mesh_features, list):
        for level_index, mesh_tensor in enumerate(mesh_features):
            report += check_mesh_features(
                f"mesh_features[{level_index}]",
                mesh_tensor,
                "3. File content requirements (Mesh Nodes)",
            )

    spec_text += textwrap.dedent(
        """\
    ### Edges

    #### Edge indices

    The following edge index files MUST be defined:

    - `g2m_edge_index.pt`
    - `m2g_edge_index.pt`
    - `m2m_edge_index.pt`
    - `mesh_up_edge_index.pt` (hierarchical graphs only, `L > 1`)
    - `mesh_down_edge_index.pt` (hierarchical graphs only, `L > 1`)

    `g2m_edge_index.pt` and `m2g_edge_index.pt` MUST each contain a single tensor  # noqa: E501
    with shape `[2, E]`, where `E` is the number of edges in that component.

    `m2m_edge_index.pt` MUST contain a list of tensors of length `L`, i.e. one
    edge-index tensor per mesh level. Each entry MUST have shape `[2, E_level]`,
    where `E_level` is the number of edges at that level.

    For hierarchical graphs, `mesh_up_edge_index.pt` and
    `mesh_down_edge_index.pt` MUST each contain a list of length `L - 1` of
    tensors, i.e. one per inter-level connection, so that entry `i` connects level  # noqa: E501
    `i` and level `i+1`. Each entry MUST have shape `[2, E_interlevel]`, where
    `E_interlevel` is the number of edges going either up
    (`mesh_up_edge_index.pt`) or down (`mesh_down_edge_index.pt`) between that
    level pair.

    For every edge-index tensor above:

    - Row `0` MUST be sender node index, row `1` MUST be receiver node index.
    - Dtype MUST be `torch.int64`.
    """
    )

    report += check_list_type_and_length(
        m2m_edge_index,
        "m2m_edge_index.pt",
        props.num_levels,
        "3. File content requirements (Edges)",
    )

    level_offsets: list[int] = []
    cumulative = 0
    for n_level_nodes in props.num_mesh_nodes_per_level:
        level_offsets.append(cumulative)
        cumulative += n_level_nodes

    if isinstance(m2m_edge_index, list):
        for level_index, level_edge_index in enumerate(m2m_edge_index):
            expected_range = None
            if level_index < len(level_offsets):
                start = level_offsets[level_index]
                stop = start + props.num_mesh_nodes_per_level[level_index]
                expected_range = (start, stop)
            report += check_edge_index(
                f"m2m_edge_index[{level_index}]",
                level_edge_index,
                "3. File content requirements (Edges)",
                expected_sender_range=expected_range,
                expected_receiver_range=expected_range,
            )

    if props.hierarchical:
        expected_len = props.num_levels - 1
        report += check_list_type_and_length(
            mesh_up_edge_index,
            "mesh_up_edge_index.pt",
            expected_len,
            "3. File content requirements (Edges)",
        )
        report += check_list_type_and_length(
            mesh_down_edge_index,
            "mesh_down_edge_index.pt",
            expected_len,
            "3. File content requirements (Edges)",
        )

        if isinstance(mesh_up_edge_index, list) and isinstance(
            mesh_down_edge_index, list
        ):
            for level_index in range(expected_len):
                if level_index + 1 >= len(props.num_mesh_nodes_per_level):
                    continue
                lower_start = level_offsets[level_index]
                lower_stop = (
                    lower_start + props.num_mesh_nodes_per_level[level_index]
                )
                upper_start = level_offsets[level_index + 1]
                upper_stop = (
                    upper_start
                    + props.num_mesh_nodes_per_level[level_index + 1]
                )

                report += check_edge_index(
                    f"mesh_up_edge_index[{level_index}]",
                    mesh_up_edge_index[level_index],
                    "3. File content requirements (Edges)",
                    expected_sender_range=(lower_start, lower_stop),
                    expected_receiver_range=(upper_start, upper_stop),
                )
                report += check_edge_index(
                    f"mesh_down_edge_index[{level_index}]",
                    mesh_down_edge_index[level_index],
                    "3. File content requirements (Edges)",
                    expected_sender_range=(upper_start, upper_stop),
                    expected_receiver_range=(lower_start, lower_stop),
                )

    report += check_edge_index(
        "g2m_edge_index", g2m_edge_index, "3. File content requirements (Edges)"
    )
    report += check_edge_index(
        "m2g_edge_index", m2g_edge_index, "3. File content requirements (Edges)"
    )

    report += check_grid_node_relationships(
        g2m_edge_index,
        m2g_edge_index,
        props.num_mesh_nodes_per_level,
        props.num_mesh_nodes_total,
        props.num_grid_nodes,
        "3. File content requirements (Edges)",
    )

    spec_text += textwrap.dedent(
        """\
    #### Edge features

    The following edge feature files MUST be defined:

    - `g2m_features.pt`
    - `m2g_features.pt`
    - `m2m_features.pt`
    - `mesh_up_features.pt` (hierarchical graphs only, `L > 1`)
    - `mesh_down_features.pt` (hierarchical graphs only, `L > 1`)

    `g2m_features.pt` and `m2g_features.pt` MUST each contain a single tensor with  # noqa: E501
    shape `[E, N_f]`, where `E` matches the number of edges in the corresponding
    `*_edge_index.pt` file.

    `m2m_features.pt` MUST contain a list of length `L`, i.e. one feature tensor
    per mesh level. Entry `i` MUST have shape `[E_level, N_f]`, where `E_level`
    matches the edge count in entry `i` of `m2m_edge_index.pt`.

    For hierarchical graphs, `mesh_up_features.pt` and `mesh_down_features.pt`
    MUST each contain a list of length `L - 1`, i.e. one feature tensor per
    inter-level connection between level `i` and `i+1`. Entry `i` MUST have shape  # noqa: E501
    `[E_interlevel, N_f]`, where `E_interlevel` matches the edge count in entry `i`  # noqa: E501
    of the corresponding `mesh_*_edge_index.pt` file.

    For every edge feature tensor above:

    - The shape MUST be `[E_component, N_f]`.
    - `N_f` MUST be exactly `3` (for 2D edges) or exactly `4` (for 3D edges). The value of `N_f` MUST be consistent across all edge feature tensors in the graph.  # noqa: E501
    - The first column (`<feature_tensor>[:, 0]`) MUST contain the total edge length (e.g., the Euclidean distance between the sender and receiver nodes).  # noqa: E501
    - The following columns MUST contain the Cartesian coordinate displacements (`vdiff = receiver_pos - sender_pos`). For 2D edges (`N_f == 3`), columns `1` and `2` are the x- and y-displacements respectively. For 3D edges (`N_f == 4`), columns `1`, `2`, and `3` are the x-, y-, and z-displacements respectively.  # noqa: E501
    - Edge features SHOULD NOT be normalized. Instead, normalization will be performed inside `neural-lam` after graph loading.  # noqa: E501
    - Dtype MUST be `torch.float32`.

    *NOTE*: The reason for requiring that the first column be the total edge length is that `neural-lam` uses this column to compute the normalization factor (the longest edge length found across the `m2m` edge features). Since edge length and the Cartesian displacements all measure distance and share the same physical scale, all edge feature columns are normalized jointly by this single factor after loading.  # noqa: E501
    """
    )

    report += check_list_type_and_length(
        m2m_features,
        "m2m_features.pt",
        props.num_levels,
        "3. File content requirements (Edge Features)",
    )

    if isinstance(m2m_features, list) and isinstance(m2m_edge_index, list):
        for level_index, level_features in enumerate(m2m_features):
            edge_feature_tensors.append(
                (f"m2m_features[{level_index}]", level_features)
            )
            expected_num = (
                m2m_edge_index[level_index].shape[1]
                if isinstance(m2m_edge_index[level_index], torch.Tensor)
                and m2m_edge_index[level_index].ndim == 2
                else None
            )
            report += check_edge_features(
                f"m2m_features[{level_index}]",
                level_features,
                expected_num,
                "3. File content requirements (Edge Features)",
            )

    if props.hierarchical:
        expected_len = props.num_levels - 1
        report += check_list_type_and_length(
            mesh_up_features,
            "mesh_up_features.pt",
            expected_len,
            "3. File content requirements (Edge Features)",
        )
        report += check_list_type_and_length(
            mesh_down_features,
            "mesh_down_features.pt",
            expected_len,
            "3. File content requirements (Edge Features)",
        )

        if isinstance(mesh_up_features, list) and isinstance(
            mesh_up_edge_index, list
        ):
            for level_index, level_features in enumerate(mesh_up_features):
                edge_feature_tensors.append(
                    (f"mesh_up_features[{level_index}]", level_features)
                )
                expected_num = None
                if (
                    level_index < len(mesh_up_edge_index)
                    and isinstance(
                        mesh_up_edge_index[level_index], torch.Tensor
                    )
                    and mesh_up_edge_index[level_index].ndim == 2
                ):
                    expected_num = mesh_up_edge_index[level_index].shape[1]
                report += check_edge_features(
                    f"mesh_up_features[{level_index}]",
                    level_features,
                    expected_num,
                    "3. File content requirements (Edge Features)",
                )

        if isinstance(mesh_down_features, list) and isinstance(
            mesh_down_edge_index, list
        ):
            for level_index, level_features in enumerate(mesh_down_features):
                edge_feature_tensors.append(
                    (f"mesh_down_features[{level_index}]", level_features)
                )
                expected_num = None
                if (
                    level_index < len(mesh_down_edge_index)
                    and isinstance(
                        mesh_down_edge_index[level_index], torch.Tensor
                    )
                    and mesh_down_edge_index[level_index].ndim == 2
                ):
                    expected_num = mesh_down_edge_index[level_index].shape[1]
                report += check_edge_features(
                    f"mesh_down_features[{level_index}]",
                    level_features,
                    expected_num,
                    "3. File content requirements (Edge Features)",
                )

    edge_feature_tensors.append(("g2m_features", g2m_features))
    expected_num_g2m = (
        g2m_edge_index.shape[1]
        if isinstance(g2m_edge_index, torch.Tensor) and g2m_edge_index.ndim == 2
        else None
    )
    report += check_edge_features(
        "g2m_features",
        g2m_features,
        expected_num_g2m,
        "3. File content requirements (Edge Features)",
    )

    edge_feature_tensors.append(("m2g_features", m2g_features))
    expected_num_m2g = (
        m2g_edge_index.shape[1]
        if isinstance(m2g_edge_index, torch.Tensor) and m2g_edge_index.ndim == 2
        else None
    )
    report += check_edge_features(
        "m2g_features",
        m2g_features,
        expected_num_m2g,
        "3. File content requirements (Edge Features)",
    )

    report += check_edge_feature_dim_consistency(
        edge_feature_tensors, "3. File content requirements (Edge Features)"
    )

    return report, spec_text, props


def cli(input_args=None):
    """
    Command-line entry point for graph validation.

    Parameters
    ----------
    input_args : list of str, optional
        Optional argument list. If `None`, arguments are read from `sys.argv`.

    Returns
    -------
    int
        Process exit code (`0` if validation passes, `1` otherwise).
    """
    parser = ArgumentParser(
        description="Validate on-disk neural-lam graph components and view the spec",  # noqa: E501
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "graph_dir",
        type=str,
        nargs="?",
        help="Path to a graph directory to validate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--print-spec-markdown",
        action="store_true",
        help="Print the graph storage specification as Markdown and exit.",
    )
    args = parser.parse_args(input_args)

    if args.print_spec_markdown:
        with skip_all_checks():
            _, spec_text, _ = validate_graph_directory(None)
        print(spec_text)
        return 0

    if not args.graph_dir:
        parser.print_help()
        return 1

    report, spec_text, props = validate_graph_directory(args.graph_dir)

    if args.json:
        payload = {
            "ok": report.ok,
            "results": [asdict(r) for r in report.results],
            "graph_properties": asdict(props),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Graph directory: {args.graph_dir}")
        print(f"Hierarchical: {props.hierarchical}")
        print(f"Mesh levels: {props.num_levels}")
        print(f"Mesh nodes per level: {props.num_mesh_nodes_per_level}")
        print(f"Mesh nodes total: {props.num_mesh_nodes_total}")
        print(f"Grid nodes (inferred): {props.num_grid_nodes}")
        print("-" * 40)
        report.console_print()

    if report.ok and not report.has_fails():
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
