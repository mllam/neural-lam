# Standard library
import sys
from unittest.mock import MagicMock, call, patch

# Third-party
import pytest


@pytest.fixture
def logger_instance():
    """Return a CustomMLFlowLogger with __init__ fully bypassed."""
    with patch.object(
        __import__(
            "neural_lam.custom_loggers", fromlist=["CustomMLFlowLogger"]
        ).CustomMLFlowLogger,
        "__init__",
        return_value=None,
    ):
        from neural_lam.custom_loggers import CustomMLFlowLogger

        instance = CustomMLFlowLogger.__new__(CustomMLFlowLogger)
        return instance


def _make_figs(n):
    """Return n mock matplotlib figures."""
    return [MagicMock(name=f"fig_{i}") for i in range(n)]


@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_single_image_uses_original_key(mock_open, mock_mlflow, logger_instance):
    """A single-figure list logs under the bare key with no index suffix."""
    figs = _make_figs(1)
    mock_img = MagicMock()
    mock_open.return_value = mock_img

    logger_instance.log_image("loss", figs)

    figs[0].savefig.assert_called_once_with("loss.png")
    mock_open.assert_called_once_with("loss.png")
    mock_mlflow.assert_called_once_with(mock_img, "loss.png")


@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_multiple_images_use_indexed_keys(mock_open, mock_mlflow, logger_instance):
    """Each figure in a multi-figure list is logged under key_0, key_1, …"""
    figs = _make_figs(3)
    mock_open.side_effect = [MagicMock(), MagicMock(), MagicMock()]

    logger_instance.log_image("val", figs)

    assert figs[0].savefig.call_args == call("val_0.png")
    assert figs[1].savefig.call_args == call("val_1.png")
    assert figs[2].savefig.call_args == call("val_2.png")

    logged_keys = [c.args[1] for c in mock_mlflow.call_args_list]
    assert logged_keys == ["val_0.png", "val_1.png", "val_2.png"]


@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_step_is_appended_to_key(mock_open, mock_mlflow, logger_instance):
    """When step is provided it is appended to the key before indexing."""
    figs = _make_figs(2)
    mock_open.side_effect = [MagicMock(), MagicMock()]

    logger_instance.log_image("metric", figs, step=5)

    save_calls = [c.args[0] for fig in figs for c in fig.savefig.call_args_list]
    assert save_calls == ["metric_5_0.png", "metric_5_1.png"]

    logged_keys = [c.args[1] for c in mock_mlflow.call_args_list]
    assert logged_keys == ["metric_5_0.png", "metric_5_1.png"]


@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_no_credentials_error_exits(mock_open, mock_mlflow, logger_instance):
    """NoCredentialsError triggers sys.exit(1) for the failing figure."""
    from botocore.exceptions import NoCredentialsError

    figs = _make_figs(1)
    mock_open.return_value = MagicMock()
    mock_mlflow.side_effect = NoCredentialsError

    with pytest.raises(SystemExit) as exc_info:
        logger_instance.log_image("err", figs)

    assert exc_info.value.code == 1


@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_no_credentials_error_exits_on_second_figure(
    mock_open, mock_mlflow, logger_instance
):
    """NoCredentialsError on any iteration (not just the first) exits."""
    from botocore.exceptions import NoCredentialsError

    figs = _make_figs(3)
    mock_open.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    mock_mlflow.side_effect = [None, NoCredentialsError, None]

    with pytest.raises(SystemExit) as exc_info:
        logger_instance.log_image("err", figs)

    assert exc_info.value.code == 1
    # Only 2 mlflow calls should have been made before the exit
    assert mock_mlflow.call_count == 2
