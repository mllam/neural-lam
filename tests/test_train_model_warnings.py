# Standard library
from unittest.mock import MagicMock, patch

# Third-party
import loguru
import pytest

# Mock loguru.logger.catch before importing train_model
loguru.logger.catch = lambda f: f

# First-party
from neural_lam.train_model import main  # noqa: E402


@pytest.mark.parametrize(
    "eval_val,load_val,expect_warning",
    [
        ("val", None, True),
        ("val", "path/to/checkpt", False),
        (None, None, False),
    ],
)
def test_eval_without_load_warning(eval_val, load_val, expect_warning):
    mock_args = MagicMock()
    mock_args.eval = eval_val
    mock_args.load = load_val
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.train_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10

    with patch(
        "neural_lam.train_model.ArgumentParser.parse_args",
        return_value=mock_args,
    ):
        with patch(
            "neural_lam.train_model.load_config_and_datastore",
            side_effect=SystemExit(0),
        ):
            with patch("neural_lam.train_model.logger.warning") as mock_warning:
                with pytest.raises(SystemExit):
                    main()
                if expect_warning:
                    mock_warning.assert_called_once()
                    assert "--load" in mock_warning.call_args[0][0]
                else:
                    mock_warning.assert_not_called()


def test_create_gif_forwarded_to_forecaster_module():
    """--create_gif must be forwarded to ForecasterModule.__init__."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = [1]
    mock_args.train_steps_to_log = [2]
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10
    mock_args.create_gif = True
    mock_args.devices = ["auto"]
    mock_args.model = "graph_lam"

    captured_kwargs = {}

    def capture_init(_self, **kwargs):
        captured_kwargs.update(kwargs)
        # Raise so we don't need to mock trainer.fit
        raise SystemExit(0)

    with (
        patch(
            "neural_lam.train_model.ArgumentParser.parse_args",
            return_value=mock_args,
        ),
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock()}),
        patch("neural_lam.train_model.ARForecaster"),
        patch(
            "neural_lam.models.module.ForecasterModule.__init__",
            capture_init,
        ),
        pytest.raises(SystemExit),
    ):
        main()

    assert (
        "create_gif" in captured_kwargs
    ), "create_gif was not forwarded to ForecasterModule"
    assert captured_kwargs["create_gif"] is True
    assert (
        "train_steps_to_log" in captured_kwargs
    ), "train_steps_to_log was not forwarded to ForecasterModule"
    assert captured_kwargs["train_steps_to_log"] == [2]


def test_train_steps_to_log_validation():
    """ValueError must be raised if steps exceed ar_steps_train."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.train_steps_to_log = [15]  # 15 > 10 (ar_steps_train)
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10

    with patch(
        "neural_lam.train_model.ArgumentParser.parse_args",
        return_value=mock_args,
    ):
        with pytest.raises(ValueError, match="Can not log training step 15"):
            main()
