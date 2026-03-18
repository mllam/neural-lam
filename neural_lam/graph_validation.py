from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class GraphValidationError:
    message: str
    file: str | None = None
    level: int | None = None

    def format(self) -> str:
        parts: list[str] = []
        if self.file is not None:
            parts.append(self.file)
        if self.level is not None:
            parts.append(f"level={self.level}")
        prefix = f"[{', '.join(parts)}] " if parts else ""
        return f"{prefix}{self.message}"


def _torch_load(path: Path):
    # Keep behaviour consistent with utils.load_graph (uses weights_only=True).
    # This is supported in the project's supported torch versions.
    return torch.load(path, weights_only=True)


def _ensure_tensor_2_by_n(x, *, file: str) -> list[GraphValidationError]:
    errors: list[GraphValidationError] = []
    if not isinstance(x, torch.Tensor):
        errors.append(
            GraphValidationError("expected a torch.Tensor", file=file)
        )
        return errors
    if x.ndim != 2 or x.shape[0] != 2:
        errors.append(
            GraphValidationError(
                f"expected shape [2, N], got {tuple(x.shape)}", file=file
            )
        )
    return errors


def _ensure_features_n_by_d(
    x, *, file: str, expected_d: Optional[int] = None
) -> list[GraphValidationError]:
    errors: list[GraphValidationError] = []
    if not isinstance(x, torch.Tensor):
        errors.append(
            GraphValidationError("expected a torch.Tensor", file=file)
        )
        return errors
    if x.ndim != 2:
        errors.append(
            GraphValidationError(
                f"expected shape [N, d], got {tuple(x.shape)}", file=file
            )
        )
        return errors
    if expected_d is not None and x.shape[1] != expected_d:
        errors.append(
            GraphValidationError(
                f"expected d={expected_d}, got d={x.shape[1]}",
                file=file,
            )
        )
    return errors


def _ensure_list_of_tensors(
    x, *, file: str
) -> tuple[list[torch.Tensor], list[GraphValidationError]]:
    errors: list[GraphValidationError] = []
    if not isinstance(x, list):
        return [], [GraphValidationError("expected a list", file=file)]
    tensors: list[torch.Tensor] = []
    for i, item in enumerate(x):
        if not isinstance(item, torch.Tensor):
            errors.append(
                GraphValidationError(
                    "expected list item to be a torch.Tensor",
                    file=file,
                    level=i,
                )
            )
            continue
        tensors.append(item)
    return tensors, errors


def validate_graph_dir(
    graph_dir: str | Path,
    *,
    expected_hierarchical: Optional[bool] = None,
    expected_n_levels: Optional[int] = None,
    expected_d_edge_features: Optional[int] = None,
    expected_d_mesh_features: Optional[int] = None,
) -> list[GraphValidationError]:
    """
    Validate a graph directory produced by neural-lam graph generation.

    Returns a list of validation errors. An empty list means the graph is OK.
    """
    graph_dir_path = Path(graph_dir)
    errors: list[GraphValidationError] = []

    required_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]

    hierarchical_files = [
        "mesh_up_edge_index.pt",
        "mesh_down_edge_index.pt",
        "mesh_up_features.pt",
        "mesh_down_features.pt",
    ]

    # Heuristic: hierarchical graphs include mesh_up/down files.
    has_hierarchical_files = all(
        (graph_dir_path / fn).exists() for fn in hierarchical_files
    )
    is_hierarchical = has_hierarchical_files
    if expected_hierarchical is not None and is_hierarchical != expected_hierarchical:
        errors.append(
            GraphValidationError(
                f"hierarchical={is_hierarchical}, expected {expected_hierarchical}"
            )
        )

    if is_hierarchical:
        required_files = required_files + hierarchical_files

    for fn in required_files:
        if not (graph_dir_path / fn).exists():
            errors.append(GraphValidationError("missing file", file=fn))

    if errors:
        # If required files are missing, stop early to avoid noisy load errors.
        return errors

    # Load files
    g2m_edge_index = _torch_load(graph_dir_path / "g2m_edge_index.pt")
    m2g_edge_index = _torch_load(graph_dir_path / "m2g_edge_index.pt")
    g2m_features = _torch_load(graph_dir_path / "g2m_features.pt")
    m2g_features = _torch_load(graph_dir_path / "m2g_features.pt")
    m2m_edge_index = _torch_load(graph_dir_path / "m2m_edge_index.pt")
    m2m_features = _torch_load(graph_dir_path / "m2m_features.pt")
    mesh_features = _torch_load(graph_dir_path / "mesh_features.pt")

    # Basic shape checks (tensors)
    errors += _ensure_tensor_2_by_n(g2m_edge_index, file="g2m_edge_index.pt")
    errors += _ensure_tensor_2_by_n(m2g_edge_index, file="m2g_edge_index.pt")
    errors += _ensure_features_n_by_d(
        g2m_features,
        file="g2m_features.pt",
        expected_d=expected_d_edge_features,
    )
    errors += _ensure_features_n_by_d(
        m2g_features,
        file="m2g_features.pt",
        expected_d=expected_d_edge_features,
    )

    # Basic type/shape checks (lists)
    m2m_edge_list, list_errs = _ensure_list_of_tensors(
        m2m_edge_index, file="m2m_edge_index.pt"
    )
    errors += list_errs
    m2m_feat_list, list_errs = _ensure_list_of_tensors(
        m2m_features, file="m2m_features.pt"
    )
    errors += list_errs
    mesh_feat_list, list_errs = _ensure_list_of_tensors(
        mesh_features, file="mesh_features.pt"
    )
    errors += list_errs

    if expected_n_levels is not None and len(m2m_edge_list) != expected_n_levels:
        errors.append(
            GraphValidationError(
                f"expected {expected_n_levels} levels, got {len(m2m_edge_list)}",
                file="m2m_edge_index.pt",
            )
        )

    if len(m2m_edge_list) != len(m2m_feat_list):
        errors.append(
            GraphValidationError(
                "level count mismatch between edge_index and features",
                file="m2m_*",
            )
        )
    if len(mesh_feat_list) != len(m2m_edge_list) and len(mesh_feat_list) != 0:
        errors.append(
            GraphValidationError(
                "level count mismatch between mesh_features and m2m_edge_index",
                file="mesh_features.pt",
            )
        )

    # Per-level checks
    for i, ei in enumerate(m2m_edge_list):
        errors += [
            e
            for e in _ensure_tensor_2_by_n(ei, file="m2m_edge_index.pt")
            if e.level is None
        ]
        if i < len(m2m_feat_list):
            feat = m2m_feat_list[i]
            errors += [
                GraphValidationError(err.message, file="m2m_features.pt", level=i)
                for err in _ensure_features_n_by_d(
                    feat,
                    file="m2m_features.pt",
                    expected_d=expected_d_edge_features,
                )
            ]

            # Edge count consistency
            if isinstance(ei, torch.Tensor) and isinstance(feat, torch.Tensor):
                if ei.ndim == 2 and feat.ndim == 2 and ei.shape[0] == 2:
                    if ei.shape[1] != feat.shape[0]:
                        errors.append(
                            GraphValidationError(
                                f"edge count mismatch: edge_index has {ei.shape[1]} edges, features has {feat.shape[0]} rows",
                                file="m2m_*",
                                level=i,
                            )
                        )

    # Mesh feature dim (optional)
    if expected_d_mesh_features is not None:
        for i, mf in enumerate(mesh_feat_list):
            if mf.ndim != 2 or mf.shape[1] != expected_d_mesh_features:
                errors.append(
                    GraphValidationError(
                        f"expected d={expected_d_mesh_features}, got shape {tuple(mf.shape)}",
                        file="mesh_features.pt",
                        level=i,
                    )
                )

    # g2m/m2g edge count consistency
    if (
        isinstance(g2m_edge_index, torch.Tensor)
        and isinstance(g2m_features, torch.Tensor)
        and g2m_edge_index.ndim == 2
        and g2m_features.ndim == 2
        and g2m_edge_index.shape[0] == 2
    ):
        if g2m_edge_index.shape[1] != g2m_features.shape[0]:
            errors.append(
                GraphValidationError(
                    f"edge count mismatch: edge_index has {g2m_edge_index.shape[1]} edges, features has {g2m_features.shape[0]} rows",
                    file="g2m_*",
                )
            )

    if (
        isinstance(m2g_edge_index, torch.Tensor)
        and isinstance(m2g_features, torch.Tensor)
        and m2g_edge_index.ndim == 2
        and m2g_features.ndim == 2
        and m2g_edge_index.shape[0] == 2
    ):
        if m2g_edge_index.shape[1] != m2g_features.shape[0]:
            errors.append(
                GraphValidationError(
                    f"edge count mismatch: edge_index has {m2g_edge_index.shape[1]} edges, features has {m2g_features.shape[0]} rows",
                    file="m2g_*",
                )
            )

    if is_hierarchical:
        up_edge_index = _torch_load(graph_dir_path / "mesh_up_edge_index.pt")
        down_edge_index = _torch_load(graph_dir_path / "mesh_down_edge_index.pt")
        up_features = _torch_load(graph_dir_path / "mesh_up_features.pt")
        down_features = _torch_load(graph_dir_path / "mesh_down_features.pt")

        up_ei_list, list_errs = _ensure_list_of_tensors(
            up_edge_index, file="mesh_up_edge_index.pt"
        )
        errors += list_errs
        up_f_list, list_errs = _ensure_list_of_tensors(
            up_features, file="mesh_up_features.pt"
        )
        errors += list_errs
        down_ei_list, list_errs = _ensure_list_of_tensors(
            down_edge_index, file="mesh_down_edge_index.pt"
        )
        errors += list_errs
        down_f_list, list_errs = _ensure_list_of_tensors(
            down_features, file="mesh_down_features.pt"
        )
        errors += list_errs

        for name, ei_list, f_list in [
            ("mesh_up", up_ei_list, up_f_list),
            ("mesh_down", down_ei_list, down_f_list),
        ]:
            if len(ei_list) != len(f_list):
                errors.append(
                    GraphValidationError(
                        "level count mismatch between edge_index and features",
                        file=f"{name}_*",
                    )
                )
            for i, (ei, f) in enumerate(zip(ei_list, f_list)):
                if ei.ndim == 2 and ei.shape[0] == 2 and f.ndim == 2:
                    if ei.shape[1] != f.shape[0]:
                        errors.append(
                            GraphValidationError(
                                f"edge count mismatch: edge_index has {ei.shape[1]} edges, features has {f.shape[0]} rows",
                                file=f"{name}_*",
                                level=i,
                            )
                        )
                if (
                    expected_d_edge_features is not None
                    and isinstance(f, torch.Tensor)
                    and f.ndim == 2
                    and f.shape[1] != expected_d_edge_features
                ):
                    errors.append(
                        GraphValidationError(
                            f"expected d={expected_d_edge_features}, got d={f.shape[1]}",
                            file=f"{name}_features.pt",
                            level=i,
                        )
                    )

    return errors

