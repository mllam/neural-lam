# Standard library
from unittest.mock import MagicMock, patch

# Third-party
import pytest


def test_restore_opt_without_load_raises():
    """
    Using --restore_opt without --load should fail the assertion
    that a checkpoint path is required for restoring training state.
    """
    # Standard library
    from argparse import Namespace

    args = Namespace(load=None, restore_opt=True)
    with pytest.raises(AssertionError, match="not loading a checkpoint"):
        assert args.load or not args.restore_opt, (
            "Can not restore training state " "when not loading a checkpoint"
        )


@patch("neural_lam.train_model.pl.Trainer")
@patch("neural_lam.train_model.WeatherDataModule")
@patch("neural_lam.train_model.load_config_and_datastore")
@patch("neural_lam.train_model.utils")
def test_weights_only_uses_load_from_checkpoint(
    mock_utils, mock_load_config, mock_dm, mock_trainer_cls
):
    """
    When --load is given without --restore_opt, load_from_checkpoint
    should be called and ckpt_path should be None in trainer.fit.
    """
    mock_config = MagicMock()
    mock_datastore = MagicMock()
    mock_datastore.root_path = "/tmp/fake"
    mock_load_config.return_value = (mock_config, mock_datastore)
    mock_utils.setup_training_logger.return_value = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer_cls.return_value = mock_trainer

    fake_ckpt = "/tmp/fake_checkpoint.ckpt"

    with patch(
        "neural_lam.train_model.MODELS",
        {"graph_lam": MagicMock()},
    ) as mock_models:
        mock_model_cls = mock_models["graph_lam"]
        mock_model = MagicMock()
        mock_model_cls.load_from_checkpoint.return_value = mock_model

        # First-party
        from neural_lam.train_model import main

        main(
            [
                "--config_path",
                "fake_config.yaml",
                "--load",
                fake_ckpt,
            ]
        )

        # Should use load_from_checkpoint for weights-only
        mock_model_cls.load_from_checkpoint.assert_called_once()
        call_args = mock_model_cls.load_from_checkpoint.call_args
        assert call_args[0][0] == fake_ckpt

        # Should NOT call the regular constructor
        mock_model_cls.assert_not_called()

        # trainer.fit should be called with ckpt_path=None
        mock_trainer.fit.assert_called_once()
        fit_kwargs = mock_trainer.fit.call_args
        assert fit_kwargs.kwargs.get("ckpt_path") is None


@patch("neural_lam.train_model.pl.Trainer")
@patch("neural_lam.train_model.WeatherDataModule")
@patch("neural_lam.train_model.load_config_and_datastore")
@patch("neural_lam.train_model.utils")
def test_restore_opt_passes_ckpt_path(
    mock_utils, mock_load_config, mock_dm, mock_trainer_cls
):
    """
    When --load and --restore_opt are both given, the model should be
    created with the normal constructor and ckpt_path should be passed
    to trainer.fit.
    """
    mock_config = MagicMock()
    mock_datastore = MagicMock()
    mock_datastore.root_path = "/tmp/fake"
    mock_load_config.return_value = (mock_config, mock_datastore)
    mock_utils.setup_training_logger.return_value = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer_cls.return_value = mock_trainer

    fake_ckpt = "/tmp/fake_checkpoint.ckpt"

    with patch(
        "neural_lam.train_model.MODELS",
        {"graph_lam": MagicMock()},
    ) as mock_models:
        mock_model_cls = mock_models["graph_lam"]
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        # First-party
        from neural_lam.train_model import main

        main(
            [
                "--config_path",
                "fake_config.yaml",
                "--load",
                fake_ckpt,
                "--restore_opt",
            ]
        )

        # Should use the normal constructor, NOT load_from_checkpoint
        mock_model_cls.assert_called_once()
        mock_model_cls.load_from_checkpoint.assert_not_called()

        # trainer.fit should be called with ckpt_path set
        mock_trainer.fit.assert_called_once()
        fit_kwargs = mock_trainer.fit.call_args
        assert fit_kwargs.kwargs.get("ckpt_path") == fake_ckpt


@patch("neural_lam.train_model.pl.Trainer")
@patch("neural_lam.train_model.WeatherDataModule")
@patch("neural_lam.train_model.load_config_and_datastore")
@patch("neural_lam.train_model.utils")
def test_eval_always_passes_ckpt_path(
    mock_utils, mock_load_config, mock_dm, mock_trainer_cls
):
    """
    When --eval is used with --load, ckpt_path should always be passed
    to trainer.test regardless of --restore_opt.
    """
    mock_config = MagicMock()
    mock_datastore = MagicMock()
    mock_datastore.root_path = "/tmp/fake"
    mock_load_config.return_value = (mock_config, mock_datastore)
    mock_utils.setup_training_logger.return_value = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer_cls.return_value = mock_trainer

    fake_ckpt = "/tmp/fake_checkpoint.ckpt"

    with patch(
        "neural_lam.train_model.MODELS",
        {"graph_lam": MagicMock()},
    ) as mock_models:
        mock_model_cls = mock_models["graph_lam"]
        mock_model = MagicMock()
        mock_model_cls.load_from_checkpoint.return_value = mock_model

        # First-party
        from neural_lam.train_model import main

        main(
            [
                "--config_path",
                "fake_config.yaml",
                "--load",
                fake_ckpt,
                "--eval",
                "test",
            ]
        )

        # trainer.test should be called with ckpt_path
        mock_trainer.test.assert_called_once()
        test_kwargs = mock_trainer.test.call_args
        assert test_kwargs.kwargs.get("ckpt_path") == fake_ckpt


@patch("neural_lam.train_model.pl.Trainer")
@patch("neural_lam.train_model.WeatherDataModule")
@patch("neural_lam.train_model.load_config_and_datastore")
@patch("neural_lam.train_model.utils")
def test_no_load_creates_fresh_model(
    mock_utils, mock_load_config, mock_dm, mock_trainer_cls
):
    """
    When no --load is given, the model should be created with the
    normal constructor and ckpt_path should be None.
    """
    mock_config = MagicMock()
    mock_datastore = MagicMock()
    mock_datastore.root_path = "/tmp/fake"
    mock_load_config.return_value = (mock_config, mock_datastore)
    mock_utils.setup_training_logger.return_value = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer_cls.return_value = mock_trainer

    with patch(
        "neural_lam.train_model.MODELS",
        {"graph_lam": MagicMock()},
    ) as mock_models:
        mock_model_cls = mock_models["graph_lam"]
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        # First-party
        from neural_lam.train_model import main

        main(
            [
                "--config_path",
                "fake_config.yaml",
            ]
        )

        # Should use the normal constructor
        mock_model_cls.assert_called_once()
        mock_model_cls.load_from_checkpoint.assert_not_called()

        # trainer.fit should be called with ckpt_path=None
        mock_trainer.fit.assert_called_once()
        fit_kwargs = mock_trainer.fit.call_args
        assert fit_kwargs.kwargs.get("ckpt_path") is None
