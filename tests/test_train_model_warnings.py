# Standard library
from unittest.mock import MagicMock, patch

# Third-party
import pytest

# First-party
from neural_lam.train_model import main


def test_eval_without_load_warning():
    """Test that the eval-without-load warning logic works."""
    # We patch argparse.ArgumentParser.parse_args to return our custom args
    # and we patch load_config_and_datastore to stop execution immediately
    # after the warning.

    mock_args = MagicMock()
    mock_args.eval = "val"
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

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
                mock_warning.assert_called_once()
                assert "--load" in mock_warning.call_args[0][0]


def test_eval_with_load_no_warning():
    """Test that no warning is raised when --load is provided."""
    mock_args = MagicMock()
    mock_args.eval = "val"
    mock_args.load = "path/to/checkpt"
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

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
                mock_warning.assert_not_called()


def test_no_eval_no_warning():
    """Test that no warning is raised in normal training mode (no --eval)."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

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
                mock_warning.assert_not_called()
