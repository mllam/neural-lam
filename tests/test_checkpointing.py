# Standard library
import sys
from unittest.mock import MagicMock, patch

# First-party
from neural_lam.train_model import main


@patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock()})
def test_checkpoint_callbacks():
    test_args = [
        "train_model.py",
        "--config_path",
        "dummy.yaml",
        "--epochs",
        "1",
    ]

    with patch(
        "neural_lam.train_model.load_config_and_datastore"
    ) as mock_load, patch(
        "neural_lam.train_model.utils.setup_training_logger"
    ), patch(
        "neural_lam.train_model.pl.Trainer"
    ) as mock_trainer, patch.object(
        sys, "argv", test_args
    ):
        mock_load.return_value = (MagicMock(), MagicMock())
        main()

        assert mock_trainer.call_count == 1
        _, kwargs = mock_trainer.call_args
        callbacks = kwargs.get("callbacks", [])

        assert len(callbacks) == 2
        filenames = [cb.filename for cb in callbacks]
        assert "min_val_loss" in filenames
        assert "last" in filenames

        train_cb = [cb for cb in callbacks if cb.filename == "last"][0]
        assert getattr(train_cb, "_save_on_train_epoch_end", None) is True
        assert train_cb.monitor is None
        assert getattr(train_cb, "enable_version_counter", None) is False
        assert train_cb.save_top_k == 1
