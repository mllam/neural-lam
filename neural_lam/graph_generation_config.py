"""On-disk metadata for graph generation (issue #470: ``graph_config.json``)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional

GRAPH_CONFIG_FILENAME = "graph_config.json"
GRAPH_CONFIG_SCHEMA_VERSION = 1

# Keys that must be present for ``validate_graph_dir`` metadata checks.
REQUIRED_GRAPH_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "hierarchical",
        "n_max_levels",
        "n_levels_on_disk",
        "grid_nx",
        "grid_ny",
        "neural_lam_version",
    }
)


def _package_version() -> str:
    try:
        return metadata.version("neural-lam")
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_commit_short() -> Optional[str]:
    sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT")
    if sha:
        return sha[:40] if len(sha) > 40 else sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()[:40]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def build_graph_config_record(
    *,
    hierarchical: bool,
    n_max_levels: Optional[int],
    n_levels_on_disk: int,
    mesh_levels_used: int,
    grid_nx: int,
    grid_ny: int,
    mesh_branching_children_per_axis: int,
    nlev_full_tree: int,
    nleaf: int,
    g2m_dm_scale: float,
    d_edge_features: int,
    d_mesh_features: int,
    neural_lam_version: Optional[str] = None,
    generated_at_utc: Optional[str] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the dict written to ``graph_config.json``."""
    ver = neural_lam_version if neural_lam_version is not None else _package_version()
    ts = generated_at_utc
    if ts is None:
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    gc = git_commit if git_commit is not None else _git_commit_short()
    return {
        "schema_version": GRAPH_CONFIG_SCHEMA_VERSION,
        "hierarchical": bool(hierarchical),
        "n_max_levels": n_max_levels,
        "n_levels_on_disk": int(n_levels_on_disk),
        "mesh_levels_used": int(mesh_levels_used),
        "grid_nx": int(grid_nx),
        "grid_ny": int(grid_ny),
        "mesh_tree": {
            "branching_children_per_axis": int(mesh_branching_children_per_axis),
            "nlev_full_tree": int(nlev_full_tree),
            "nleaf": int(nleaf),
        },
        "g2m": {"dm_scale": float(g2m_dm_scale)},
        "tensor_layout": {
            "d_edge_features": int(d_edge_features),
            "d_mesh_features": int(d_mesh_features),
        },
        "neural_lam_version": ver,
        "generated_at_utc": ts,
        "git_commit": gc,
    }


def write_graph_config(
    graph_dir_path: str | Path,
    record: Mapping[str, Any],
    *,
    indent: int = 2,
) -> Path:
    path = Path(graph_dir_path) / GRAPH_CONFIG_FILENAME
    path.write_text(
        json.dumps(dict(record), indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
