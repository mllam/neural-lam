# Standard library
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

    with (
        patch("neural_lam.train_model.load_config_and_datastore") as mock_load,
        patch("neural_lam.train_model.utils.setup_training_logger"),
        patch("neural_lam.train_model.pl.Trainer") as mock_trainer,
    ):
        mock_load.return_value = (MagicMock(), MagicMock())
        main(input_args=test_args[1:])

        assert mock_trainer.call_count == 1
        _, kwargs = mock_trainer.call_args
        callbacks = kwargs.get("callbacks", [])

        assert len(callbacks) == 2
        filenames = [cb.filename for cb in callbacks]
        assert "min_val_loss" in filenames
        assert "last" in filenames

        # Verify the rescue checkpoint (last.ckpt) configs
        train_cb = [cb for cb in callbacks if cb.filename == "last"][0]
        # Assert public behaviors configured in the training loop
        assert train_cb.monitor is None
        assert train_cb.save_top_k == 1

        # Verify custom overriding attributes dynamically
        # Checking string representations or checking dict attributes is safer
        # than asserting strict private types across lightning variations.
        assert (
            getattr(
                train_cb,
                "save_on_train_epoch_end",
                getattr(train_cb, "_save_on_train_epoch_end", None),
            )
            is True
        )
