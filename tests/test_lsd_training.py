
import pytest
import pytorch_lightning as pl
import torch
import wandb
from pathlib import Path
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example

def run_lsd_training(datastore):
    """
    Run one epoch of training with LSD loss.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        num_devices = 2
    else:
        num_devices = 1

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        devices=num_devices,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    graph_name = "1level_lsd"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=1,
        ar_steps_eval=1,
        standardize=True,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    class ModelArgs:
        output_std = False
        loss = "lsd"
        restore_opt = False
        n_example_pred = 0
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 1
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1]
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
    
    # Mock wandb to avoid network calls
    with torch.no_grad():
        trainer.fit(model=model, datamodule=data_module)

def test_training_lsd():
    """Test training with LSD loss on dummy data"""
    datastore = init_datastore_example("dummydata")
    run_lsd_training(datastore)
