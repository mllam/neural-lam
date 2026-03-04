# Standard library
from pathlib import Path

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


def _build_trainer(gradient_clip_val, gradient_clip_algorithm="norm"):
    """Build a minimal Trainer with specified gradient clipping settings."""
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    return pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        devices=2,
        log_every_n_steps=1,
        detect_anomaly=True,
        gradient_clip_val=(
            gradient_clip_val if gradient_clip_val > 0 else None
        ),
        gradient_clip_algorithm=gradient_clip_algorithm,
    )


def test_gradient_clipping_enabled():
    """Verify that gradient_clip_val=1.0 is correctly set on the Trainer."""
    trainer = _build_trainer(gradient_clip_val=1.0)
    assert trainer.gradient_clip_val == 1.0
    assert trainer.gradient_clip_algorithm == "norm"


def test_gradient_clipping_disabled():
    """Verify that gradient_clip_val=0 disables clipping (None)."""
    trainer = _build_trainer(gradient_clip_val=0)
    assert trainer.gradient_clip_val is None


def test_gradient_clip_algorithm_value():
    """Verify that the 'value' algorithm is properly forwarded."""
    trainer = _build_trainer(
        gradient_clip_val=1.0, gradient_clip_algorithm="value"
    )
    assert trainer.gradient_clip_algorithm == "value"


def test_gradient_clip_algorithm_norm():
    """Verify that the 'norm' algorithm is properly forwarded."""
    trainer = _build_trainer(
        gradient_clip_val=2.5, gradient_clip_algorithm="norm"
    )
    assert trainer.gradient_clip_val == 2.5
    assert trainer.gradient_clip_algorithm == "norm"


def test_training_with_gradient_clipping():
    """End-to-end: run 1 epoch with gradient clipping, assert no NaN loss."""
    datastore = init_datastore_example("dummydata")

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        devices=2,
        log_every_n_steps=1,
        detect_anomaly=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

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

    model_args = ModelArgs()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(
        args=model_args,
        datastore=datastore,
        config=config,
    )
    wandb.init()
    trainer.fit(model=model, datamodule=data_module)

    # Assert training completed without NaN loss
    assert trainer.callback_metrics.get("train_loss_epoch") is not None
    train_loss = trainer.callback_metrics["train_loss_epoch"].item()
    assert not torch.isnan(
        torch.tensor(train_loss)
    ), f"Training loss is NaN: {train_loss}"
