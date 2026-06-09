# Standard library
from unittest.mock import MagicMock, call, patch

# Third-party
import pytest

# First-party
from neural_lam.custom_loggers import CustomMLFlowLogger


@pytest.fixture
def logger_instance():
    """Return a CustomMLFlowLogger with __init__ fully bypassed."""
    instance = CustomMLFlowLogger.__new__(CustomMLFlowLogger)
    instance._save_dir = ""
    return instance


def _make_figs(n):
    """Return n mock matplotlib figures."""
    return [MagicMock(name=f"fig_{i}") for i in range(n)]


def _ctx_mock(value):
    """Return a MagicMock whose __enter__ yields ``value``."""
    m = MagicMock()
    m.__enter__.return_value = value
    return m


@patch("os.path.exists", return_value=False)
@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_single_image_uses_original_key(
    mock_open, mock_mlflow, _mock_exists, logger_instance
):
    """A single-figure list logs under the bare key with no index suffix."""
    figs = _make_figs(1)
    mock_img = MagicMock()
    mock_open.return_value = _ctx_mock(mock_img)

    logger_instance.log_image("loss", figs)

    figs[0].savefig.assert_called_once_with("loss.png")
    mock_open.assert_called_once_with("loss.png")
    mock_mlflow.assert_called_once_with(mock_img, "loss.png")


@patch("os.path.exists", return_value=False)
@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_multiple_images_use_indexed_keys(
    mock_open, mock_mlflow, _mock_exists, logger_instance
):
    """Each figure in a multi-figure list is logged under key_0, key_1, ..."""
    figs = _make_figs(3)
    mock_open.side_effect = [_ctx_mock(MagicMock()) for _ in range(3)]

    logger_instance.log_image("val", figs)

    assert figs[0].savefig.call_args == call("val_0.png")
    assert figs[1].savefig.call_args == call("val_1.png")
    assert figs[2].savefig.call_args == call("val_2.png")

    logged_keys = [c.args[1] for c in mock_mlflow.call_args_list]
    assert logged_keys == ["val_0.png", "val_1.png", "val_2.png"]


@patch("os.path.exists", return_value=False)
@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_step_is_appended_to_key(
    mock_open, mock_mlflow, _mock_exists, logger_instance
):
    """When step is provided it is appended to the key before indexing."""
    figs = _make_figs(2)
    mock_open.side_effect = [_ctx_mock(MagicMock()) for _ in range(2)]

    logger_instance.log_image("metric", figs, step=5)

    save_calls = [c.args[0] for fig in figs for c in fig.savefig.call_args_list]
    assert save_calls == ["metric_5_0.png", "metric_5_1.png"]

    logged_keys = [c.args[1] for c in mock_mlflow.call_args_list]
    assert logged_keys == ["metric_5_0.png", "metric_5_1.png"]


@patch("os.path.exists", return_value=False)
@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_no_credentials_error_propagates(
    mock_open, mock_mlflow, _mock_exists, logger_instance
):
    """NoCredentialsError is re-raised so callers can handle it."""
    # Third-party
    from botocore.exceptions import NoCredentialsError

    figs = _make_figs(1)
    mock_open.return_value = _ctx_mock(MagicMock())
    mock_mlflow.side_effect = NoCredentialsError

    with pytest.raises(NoCredentialsError):
        logger_instance.log_image("err", figs)


@patch("os.path.exists", return_value=False)
@patch("mlflow.log_image")
@patch("PIL.Image.open")
def test_no_credentials_error_propagates_on_second_figure(
    mock_open, mock_mlflow, _mock_exists, logger_instance
):
    """A NoCredentialsError on any iteration (not just the first) propagates."""
    # Third-party
    from botocore.exceptions import NoCredentialsError

    figs = _make_figs(3)
    mock_open.side_effect = [_ctx_mock(MagicMock()) for _ in range(3)]
    mock_mlflow.side_effect = [None, NoCredentialsError, None]

    with pytest.raises(NoCredentialsError):
        logger_instance.log_image("err", figs)

    assert mock_mlflow.call_count == 2
