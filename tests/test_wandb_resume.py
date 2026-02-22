# Standard library
from unittest.mock import MagicMock, patch

# Third-party
import pytest


def _parse_real(args_list):
    """Parse args through the real train_model parser."""
    # First-party
    from neural_lam.train_model import _build_arg_parser

    return _build_arg_parser().parse_args(args_list)


# --- Argument parsing tests ---------------------------------------------------


def test_wandb_id_default_none():
    args = _parse_real(["--config_path", "dummy.yaml"])
    assert args.wandb_id is None


def test_wandb_id_parsed():
    args = _parse_real(
        ["--config_path", "dummy.yaml", "--wandb_id", "abc123xyz"]
    )
    assert args.wandb_id == "abc123xyz"


def test_wandb_resume_not_exposed():
    """--wandb_resume must no longer exist as a CLI argument."""
    with pytest.raises(SystemExit):
        _parse_real(["--config_path", "dummy.yaml", "--wandb_resume", "allow"])


# --- setup_training_logger tests ----------------------------------------------


def _make_args(wandb_id=None):
    args = MagicMock()
    args.logger = "wandb"
    args.logger_project = "neural_lam"
    args.wandb_id = wandb_id
    return args


@pytest.mark.parametrize(
    "wandb_id, expected_resume, expected_id, expected_name",
    [
        (None, None, None, "my-run"),
        ("abc123", "allow", "abc123", None),
    ],
)
@patch("neural_lam.utils.pl.loggers.WandbLogger")
def test_wandb_logger_kwargs(
    mock_wandb, wandb_id, expected_resume, expected_id, expected_name
):
    """WandbLogger is called with the correct resume, id, and name kwargs."""
    # First-party
    from neural_lam.utils import setup_training_logger

    args = _make_args(wandb_id=wandb_id)
    datastore = MagicMock()
    datastore._config = {}

    setup_training_logger(datastore, args, run_name="my-run")

    _, kwargs = mock_wandb.call_args
    assert kwargs["resume"] == expected_resume
    assert kwargs["id"] == expected_id
    assert kwargs["name"] == expected_name


def test_wandb_id_ignored_with_mlflow_warns():
    """--wandb_id is ignored when logger=mlflow and a warning is emitted."""
    # First-party
    from neural_lam.utils import setup_training_logger

    args = MagicMock()
    args.logger = "mlflow"
    args.logger_project = "neural_lam"
    args.wandb_id = "abc123"

    datastore = MagicMock()
    datastore._config = {}

    with (
        patch("neural_lam.utils.CustomMLFlowLogger"),
        patch.dict(
            "os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"}
        ),
        pytest.warns(
            UserWarning, match="--wandb_id is only used with --logger=wandb"
        ),
    ):
        setup_training_logger(datastore, args, run_name="my-run")
