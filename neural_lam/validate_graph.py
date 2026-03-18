from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from loguru import logger

from .graph_validation import validate_graph_dir


def cli(input_args=None) -> int:
    parser = ArgumentParser(
        description="Validate a generated neural-lam graph directory",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Path to graph directory containing *.pt graph files",
    )
    parser.add_argument(
        "--d_edge_features",
        type=int,
        default=None,
        help="Optional: expected feature dimension for edge feature tensors",
    )
    parser.add_argument(
        "--d_mesh_features",
        type=int,
        default=None,
        help="Optional: expected feature dimension for mesh static features",
    )
    args = parser.parse_args(input_args)

    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        logger.error(f"Graph directory does not exist: {graph_dir}")
        return 2

    errors = validate_graph_dir(
        graph_dir,
        expected_d_edge_features=args.d_edge_features,
        expected_d_mesh_features=args.d_mesh_features,
    )
    if not errors:
        logger.info(f"OK: graph at {graph_dir} passed all validation checks.")
        return 0

    logger.error(
        f"Found {len(errors)} validation issue(s) in graph at {graph_dir}:"
    )
    for err in errors:
        logger.error(f"- {err.format()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())

