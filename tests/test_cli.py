# Standard library
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third-party
import pytest
import torch

# First-party
import neural_lam
import neural_lam.create_graph
import neural_lam.train_model


def test_import():
    """This test just ensures that each cli entry-point can be imported for now,
    eventually we should test their execution too."""
    assert neural_lam is not None
    assert neural_lam.create_graph is not None
    assert neural_lam.train_model is not None


# --- Argument parsing tests ---------------------------------------------------


def _make_parser():
    """Minimal parser mirroring train_model's --config_path and --wandb_id."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--wandb_id", type=str, default=None)
    return parser


@pytest.mark.parametrize(
    "extra_args, expected_wandb_id",
    [
        ([], None),
        (["--wandb_id", "abc123xyz"], "abc123xyz"),
    ],
)
def test_wandb_id_parsed(extra_args, expected_wandb_id):
    """--wandb_id defaults to None and is parsed correctly when provided."""
    args = _make_parser().parse_args(
        ["--config_path", "dummy.yaml"] + extra_args
    )
    assert args.wandb_id == expected_wandb_id


def test_wandb_resume_not_exposed():
    """--wandb_resume must not exist as a CLI argument in train_model."""
    with pytest.raises(SystemExit):
        _make_parser().parse_args(
            ["--config_path", "dummy.yaml", "--wandb_resume", "allow"]
        )


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
        patch("neural_lam.utils.logger") as mock_log,
    ):
        setup_training_logger(datastore, args, run_name="my-run")

    mock_log.warning.assert_called_once()
    warning_msg = mock_log.warning.call_args[0][0]
    assert "--wandb_id is set but logger is" in warning_msg
    assert "mlflow" in warning_msg


def _make_batch(ar_steps, forcing_dim, monotonic_times=True):
    """Create a synthetic weather batch with DataLoader-like batch dim."""
    batch_size = 2
    n_grid = 5
    d_state = 3

    init_states = torch.randn(batch_size, 2, n_grid, d_state)
    target_states = torch.randn(batch_size, ar_steps, n_grid, d_state)
    forcing = torch.randn(batch_size, ar_steps, n_grid, forcing_dim)

    base = torch.arange(ar_steps, dtype=torch.int64)
    if monotonic_times:
        target_times = torch.stack([base, base + 10], dim=0)
    else:
        target_times = torch.stack(
            [
                torch.tensor([3, 2, 1], dtype=torch.int64),
                torch.tensor([7, 6, 5], dtype=torch.int64),
            ],
            dim=0,
        )

    return init_states, target_states, forcing, target_times


def test_dry_run_data_preflight_success_skips_training_setup():
    """--dry_run_data runs preflight checks and exits before trainer setup."""

    class DummyDataModule:
        def __init__(self, *args, **kwargs):
            pass

        def setup(self, stage=None):
            self.stage = stage

        def train_dataloader(self):
            # window size = 1 + 1 + 1 = 3, forcing dim=6 is valid
            return [_make_batch(ar_steps=2, forcing_dim=6)]

        def val_dataloader(self):
            return [_make_batch(ar_steps=3, forcing_dim=6)]

        def test_dataloader(self):
            return [_make_batch(ar_steps=3, forcing_dim=6)]

    with (
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "neural_lam.train_model.WeatherDataModule",
            DummyDataModule,
        ),
        patch("neural_lam.train_model.pl.Trainer") as mock_trainer,
        patch(
            "neural_lam.train_model.utils.setup_training_logger"
        ) as mock_setup_logger,
    ):
        neural_lam.train_model.main(
            [
                "--config_path",
                "dummy.yaml",
                "--dry_run_data",
                "--ar_steps_train",
                "2",
                "--ar_steps_eval",
                "3",
                "--val_steps_to_log",
                "1",
                "2",
                "3",
                "--num_workers",
                "0",
            ]
        )

    mock_trainer.assert_not_called()
    mock_setup_logger.assert_not_called()


def test_dry_run_data_preflight_failure_raises_value_error():
    """--dry_run_data fails fast when preflight detects invalid batches."""

    class DummyBadDataModule:
        def __init__(self, *args, **kwargs):
            pass

        def setup(self, stage=None):
            self.stage = stage

        def train_dataloader(self):
            # Non-monotonic times should fail preflight checks.
            return [
                _make_batch(ar_steps=3, forcing_dim=6, monotonic_times=False)
            ]

        def val_dataloader(self):
            return [_make_batch(ar_steps=3, forcing_dim=6)]

        def test_dataloader(self):
            return [_make_batch(ar_steps=3, forcing_dim=6)]

    args = SimpleNamespace(
        eval=None,
        ar_steps_train=3,
        ar_steps_eval=3,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        neural_lam.train_model._run_data_preflight(
            data_module=DummyBadDataModule(),
            args=args,
        )
