# Standard library
from argparse import ArgumentParser
from unittest.mock import MagicMock, patch

# Third-party
import pytest


def _make_parser():
    """Reconstruct only the logger-related args from train_model's parser,
    so this test has no heavy imports and runs instantly."""
    parser = ArgumentParser()
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--logger-project", type=str, default="neural_lam")
    parser.add_argument("--logger_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    return parser


# --- Argument parsing tests ---------------------------------------------------


def test_wandb_id_default_none():
    args = _make_parser().parse_args([])
    assert args.wandb_id is None


def test_wandb_id_parsed():
    args = _make_parser().parse_args(["--wandb_id", "abc123xyz"])
    assert args.wandb_id == "abc123xyz"


def test_wandb_resume_not_exposed():
    """--wandb_resume must no longer exist as a CLI argument."""
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--wandb_resume", "allow"])


# --- setup_training_logger tests ----------------------------------------------


def _make_args(wandb_id=None):
    args = MagicMock()
    args.logger = "wandb"
    args.logger_project = "neural_lam"
    args.wandb_id = wandb_id
    return args


@patch("neural_lam.utils.pl.loggers.WandbLogger")
def test_no_id_creates_new_run(mock_wandb):
    """Without --wandb_id, resume must be None and name is forwarded."""
    from neural_lam.utils import setup_training_logger

    args = _make_args()
    datastore = MagicMock()
    datastore._config = {}

    setup_training_logger(datastore, args, run_name="my-run")

    _, kwargs = mock_wandb.call_args
    assert kwargs["resume"] is None
    assert kwargs["id"] is None
    assert kwargs["name"] == "my-run"


@patch("neural_lam.utils.pl.loggers.WandbLogger")
def test_with_id_sets_resume_allow(mock_wandb):
    """With --wandb_id, resume must be 'allow' automatically."""
    from neural_lam.utils import setup_training_logger

    args = _make_args(wandb_id="abc123")
    datastore = MagicMock()
    datastore._config = {}

    setup_training_logger(datastore, args, run_name="my-run")

    _, kwargs = mock_wandb.call_args
    assert kwargs["resume"] == "allow"
    assert kwargs["id"] == "abc123"


@patch("neural_lam.utils.pl.loggers.WandbLogger")
def test_with_id_suppresses_name(mock_wandb):
    """With --wandb_id, name must be None to preserve the existing run name."""
    from neural_lam.utils import setup_training_logger

    args = _make_args(wandb_id="abc123")
    datastore = MagicMock()
    datastore._config = {}

    setup_training_logger(datastore, args, run_name="my-run")

    _, kwargs = mock_wandb.call_args
    assert kwargs["name"] is None
