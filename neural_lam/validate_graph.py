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
    parser.add_argument(
        "--require_graph_config",
        action="store_true",
        help="Require graph_config.json to exist and be well-formed",
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
        require_graph_config=args.require_graph_config,
    )
    if not errors:
        logger.info(f"OK: graph at {graph_dir} passed all validation checks.")
        return 0

    structural = [e for e in errors if not e.message.startswith("graph_config:")]
    metadata = [e for e in errors if e.message.startswith("graph_config:")]

    if structural:
        logger.error(
            f"Found {len(structural)} structural issue(s) in graph at {graph_dir}:"
        )
        for err in structural:
            logger.error(f"- {err.format()}")
    if metadata:
        logger.error(
            f"Found {len(metadata)} graph_config issue(s) in graph at {graph_dir}:"
        )
        for err in metadata:
            logger.error(f"- {err.format()}")

    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
