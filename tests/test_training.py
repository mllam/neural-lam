# Standard library
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb
from test_datastores import DATASTORES, init_datastore

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore(datastore_name)

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=3,
        deterministic=True,
        accelerator=device_name,
        devices=1,
        log_every_n_steps=1,
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
        forcing_window_size=3,
    )

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        # XXX: this should be superfluous when we have already defined the
        # model object no?
        graph = graph_name
        hidden_dim = 8
        hidden_layers = 1
        processor_layers = 4
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1]
        metrics_watch = []

    model_args = ModelArgs()

    model = GraphLAM(  # noqa
        args=model_args,
        forcing_window_size=data_module.forcing_window_size,
        datastore=datastore,
    )
    wandb.init()
    trainer.fit(model=model, datamodule=data_module)


# def test_train_model_reduced_meps_dataset():
#     args = [
#         "--model=hi_lam",
#         "--data_config=data/meps_example_reduced/data_config.yaml",
#         "--n_workers=4",
#         "--epochs=1",
#         "--graph=hierarchical",
#         "--hidden_dim=16",
#         "--hidden_layers=1",
#         "--processor_layers=1",
#         "--ar_steps=1",
#         "--eval=val",
#         "--n_example_pred=0",
#     ]
#     train_model(args)
