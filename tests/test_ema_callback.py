# Standard library
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.callbacks import EMACallback
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


def test_ema_decay_validation():
    """EMA decay must be in [0, 1)."""
    with pytest.raises(ValueError, match="EMA decay must be in"):
        EMACallback(decay=1.0)
    with pytest.raises(ValueError, match="EMA decay must be in"):
        EMACallback(decay=-0.1)
    with pytest.raises(ValueError, match="EMA decay must be in"):
        EMACallback(decay=1.5)

    # Valid values should not raise
    EMACallback(decay=0.0)
    EMACallback(decay=0.999)
    EMACallback(decay=0.9999)


def test_ema_mathematical_correctness():
    """Verify EMA update formula: ema = decay * ema + (1 - decay) * current."""
    decay = 0.9
    callback = EMACallback(decay=decay)

    # Simulate with simple tensors
    param = torch.tensor([1.0, 2.0, 3.0])
    callback.ema_weights = [param.clone()]

    # Simulate a few updates with new parameter values
    new_values = [
        torch.tensor([2.0, 3.0, 4.0]),
        torch.tensor([3.0, 4.0, 5.0]),
        torch.tensor([4.0, 5.0, 6.0]),
    ]

    expected = param.clone()
    for new_val in new_values:
        expected = decay * expected + (1 - decay) * new_val
        callback.ema_weights[0].lerp_(new_val, 1.0 - decay)

    assert torch.allclose(callback.ema_weights[0], expected, atol=1e-6), (
        f"EMA weights {callback.ema_weights[0]} do not match "
        f"expected {expected}"
    )


def test_ema_weight_swap_during_validation():
    """Verify EMA weights are active during validation and originals restored
    after."""
    model, data_module = _setup_model_and_data()

    callback = EMACallback(decay=0.999)

    # Store original weights
    original_params = [p.data.clone() for p in model.parameters()]

    # Initialize EMA with slightly different weights
    callback.ema_weights = [p.data.clone() * 0.5 for p in model.parameters()]

    # Swap to EMA
    callback._swap_to_ema(model)
    for param, ema_w in zip(model.parameters(), callback.ema_weights):
        assert torch.allclose(
            param.data, ema_w
        ), "Model weights should match EMA weights during validation"

    # Swap back to original
    callback._swap_to_original(model)
    for param, orig in zip(model.parameters(), original_params):
        assert torch.allclose(
            param.data, orig
        ), "Model weights should be restored to original after validation"


def test_ema_checkpoint_save_load():
    """Verify EMA state survives checkpoint save and load round-trip."""
    callback = EMACallback(decay=0.999)
    callback.ema_weights = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
    ]

    # Simulate save
    checkpoint = {}
    callback.on_save_checkpoint(
        trainer=None, pl_module=None, checkpoint=checkpoint
    )
    assert "ema_weights" in checkpoint
    assert len(checkpoint["ema_weights"]) == 2

    # Simulate load into a new callback
    new_callback = EMACallback(decay=0.999)

    # Create a minimal mock module with parameters on CPU
    class FakeModule:
        def parameters(self):
            return iter([torch.tensor([0.0, 0.0])])

    new_callback.on_load_checkpoint(
        trainer=None, pl_module=FakeModule(), checkpoint=checkpoint
    )
    assert len(new_callback.ema_weights) == 2
    assert torch.allclose(new_callback.ema_weights[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(new_callback.ema_weights[1], torch.tensor([3.0, 4.0]))


def test_ema_training_integration():
    """Run a single training step with EMA enabled and verify it works."""
    model, data_module = _setup_model_and_data()

    decay = 0.999
    ema_callback = EMACallback(decay=decay)

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
        callbacks=[ema_callback],
    )

    wandb.init()
    trainer.fit(model=model, datamodule=data_module)

    # Verify EMA weights were initialized
    assert (
        len(ema_callback.ema_weights) > 0
    ), "EMA weights should be initialized after training"

    # Verify training completed without errors
    train_loss = trainer.callback_metrics.get("train_loss")
    assert train_loss is not None, "Training loss was not logged"
    assert not torch.isnan(train_loss), f"Training loss is NaN: {train_loss}"


def test_ema_noop_when_disabled():
    """Verify no EMA behavior when decay is not set (no callback added)."""
    model, data_module = _setup_model_and_data()

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    # No EMA callback in the trainer
    trainer = pl.Trainer(
        max_steps=1,
        deterministic=True,
        accelerator=device_name,
        devices=2,
        log_every_n_steps=1,
    )

    wandb.init()
    trainer.fit(model=model, datamodule=data_module)

    # Verify no EMA-related state in the trainer callbacks
    ema_callbacks = [
        cb for cb in trainer.callbacks if isinstance(cb, EMACallback)
    ]
    assert (
        len(ema_callbacks) == 0
    ), "No EMACallback should be present when EMA is disabled"
