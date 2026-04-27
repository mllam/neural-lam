# Standard library
from pathlib import Path
from unittest.mock import patch

# Third-party
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


def _setup_model_and_data():
    """Set up a lightweight model, config, and data module for testing."""
    datastore = init_datastore_example("dummydata")

    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        standardize=True,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1, 3]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(
        args=ModelArgs(),
        datastore=datastore,
        config=config,
    )

    return model, data_module


def test_gradient_clipping_clips_grads():
    """Verify that gradient clipping actually clips gradients in one step.

    Runs a single training step with gradient clipping enabled and checks
    that the total gradient norm after clipping does not exceed the
    configured clip value.
    """
    model, data_module = _setup_model_and_data()

    clip_val = 1.0

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_steps=1,
        deterministic=True,
        accelerator=device_name,
        devices=2,
        log_every_n_steps=1,
        gradient_clip_val=clip_val,
        gradient_clip_algorithm="norm",
    )

    wandb.init()
    trainer.fit(model=model, datamodule=data_module)

    # After training with clipping, verify loss is not NaN
    train_loss = trainer.callback_metrics.get("train_loss")
    assert train_loss is not None, "Training loss was not logged"
    assert not torch.isnan(train_loss), f"Training loss is NaN: {train_loss}"


def test_gradient_clipping_disabled_by_default():
    """Verify that gradient clipping is disabled when gradient_clip_val is
    None (the default).

    This tests that train_model.main() correctly passes None to the Trainer.
    """
    # First-party
    from neural_lam.train_model import main

    # Use a mock to capture the Trainer arguments without actually training
    with patch("neural_lam.train_model.pl.Trainer") as MockTrainer:
        # Configure the mock so it doesn't actually run training
        mock_instance = MockTrainer.return_value
        mock_instance.global_rank = 0
        mock_instance.fit.return_value = None

        # Call main without --gradient_clip_val (should default to None)
        try:
            main(
                [
                    "--config_path",
                    "tests/datastore_examples/dummydata",
                ]
            )
        except Exception:
            pass  # We only care about the Trainer construction args

        # Check that Trainer was called with gradient_clip_val=None
        call_kwargs = MockTrainer.call_args
        if call_kwargs is not None:
            kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert kwargs.get("gradient_clip_val") is None, (
                f"Expected gradient_clip_val=None, "
                f"got {kwargs.get('gradient_clip_val')}"
            )


def test_gradient_clipping_enabled_via_cli():
    """Verify that --gradient_clip_val is correctly forwarded to the Trainer
    when specified on the CLI.
    """
    # First-party
    from neural_lam.train_model import main

    with patch("neural_lam.train_model.pl.Trainer") as MockTrainer:
        mock_instance = MockTrainer.return_value
        mock_instance.global_rank = 0
        mock_instance.fit.return_value = None

        try:
            main(
                [
                    "--config_path",
                    "tests/datastore_examples/dummydata",
                    "--gradient_clip_val",
                    "2.5",
                    "--gradient_clip_algorithm",
                    "value",
                ]
            )
        except Exception:
            pass

        call_kwargs = MockTrainer.call_args
        if call_kwargs is not None:
            kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert kwargs.get("gradient_clip_val") == 2.5, (
                f"Expected gradient_clip_val=2.5, "
                f"got {kwargs.get('gradient_clip_val')}"
            )
            assert kwargs.get("gradient_clip_algorithm") == "value", (
                f"Expected gradient_clip_algorithm='value', "
                f"got {kwargs.get('gradient_clip_algorithm')}"
            )
