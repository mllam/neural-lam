# Standard library
import warnings
from unittest.mock import patch, MagicMock

# Third-party
import pytest

# First-party
from neural_lam.train_model import main

def test_eval_without_load_warning():
    """Test that the eval-without-load warning logic works."""
    # We patch argparse.ArgumentParser.parse_args to return our custom args
    # and we patch load_config_and_datastore to stop execution immediately after the warning.
    
    mock_args = MagicMock()
    mock_args.eval = "val"
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

    with patch("neural_lam.train_model.ArgumentParser.parse_args", return_value=mock_args):
        with patch("neural_lam.train_model.load_config_and_datastore", side_effect=SystemExit(0)):
            # Use pytest.warns to capture the actual warning from the module
            with pytest.warns(UserWarning, match="--load"):
                with pytest.raises(SystemExit):
                    main()

def test_eval_with_load_no_warning():
    """Test that no warning is raised when --load is provided."""
    mock_args = MagicMock()
    mock_args.eval = "val"
    mock_args.load = "path/to/checkpt"
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

    with patch("neural_lam.train_model.ArgumentParser.parse_args", return_value=mock_args):
        with patch("neural_lam.train_model.load_config_and_datastore", side_effect=SystemExit(0)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with pytest.raises(SystemExit):
                    main()
                # Check that no UserWarning was emitted
                user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]
                assert len(user_warnings) == 0
