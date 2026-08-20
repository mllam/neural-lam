"""High-level Python API for Neural-LAM."""

# Standard library
import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Local
from . import create_graph as create_graph_script
from . import train_model as train_model_script


@dataclass
class Run:
    """Handle containing output paths for a Neural-LAM run."""

    run_dir: Path
    checkpoint_path: Path

    @property
    def plot_dir(self) -> Path:
        """Directory containing generated plots."""
        return self.run_dir / "plots"

    @property
    def example_plots(self) -> Path:
        """Directory containing example evaluation plots."""
        return self.plot_dir / "example_plots"

    @property
    def rmse_plot(self) -> Path:
        """Path to the generated RMSE map plot."""
        return self.plot_dir / "rmse.png"


def _inject_signature(target_func: Callable, parser_func: Callable) -> None:
    """
    Update target_func's signature with arguments from the parser.
    """
    parser = parser_func()

    params = []
    for action in parser._actions:
        if action.dest == "help":
            continue

        annotation = inspect.Parameter.empty
        if action.type is not None:
            annotation = action.type
        elif action.default is not None:
            annotation = type(action.default)

        params.append(
            inspect.Parameter(
                name=action.dest,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=action.default,
                annotation=annotation,
            )
        )

    params.append(
        inspect.Parameter(
            name="config",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
        )
    )
    params.append(
        inspect.Parameter(
            name="datastore",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
        )
    )

    target_func.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=params
    )


def train(**kwargs: Any) -> Run:
    """Train a Neural-LAM model."""
    parser = train_model_script.build_parser()

    config = kwargs.pop("config", None)
    datastore = kwargs.pop("datastore", None)

    valid_args = {action.dest for action in parser._actions}
    for k in kwargs:
        if k not in valid_args:
            raise ValueError(f"Unknown argument: {k}")

    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest != "help":
            setattr(args, action.dest, action.default)

    for k, v in kwargs.items():
        setattr(args, k, v)

    return train_model_script.run(args, config=config, datastore=datastore)


def evaluate(**kwargs: Any) -> Run:
    """Evaluate a Neural-LAM model."""
    if "eval" not in kwargs:
        kwargs["eval"] = "test"
    return train(**kwargs)


def create_graph(**kwargs: Any) -> None:
    """Generate graph components."""
    parser = create_graph_script.build_parser()

    config = kwargs.pop("config", None)
    datastore = kwargs.pop("datastore", None)

    valid_args = {action.dest for action in parser._actions}
    for k in kwargs:
        if k not in valid_args:
            raise ValueError(f"Unknown argument: {k}")

    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest != "help":
            setattr(args, action.dest, action.default)

    for k, v in kwargs.items():
        setattr(args, k, v)

    create_graph_script.run(args, config=config, datastore=datastore)


def _init_signatures():
    """Initialize function signatures."""
    # Defer signature injection until called to avoid import cycles
    _inject_signature(train, train_model_script.build_parser)
    _inject_signature(evaluate, train_model_script.build_parser)
    _inject_signature(create_graph, create_graph_script.build_parser)
