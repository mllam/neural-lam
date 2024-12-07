# Standard library
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.build_rectangular_graph import build_graph_from_archetype
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import (
    DATASTORES_BOUNDARY_EXAMPLES,
    get_test_mesh_dist,
    init_datastore_boundary_example,
    init_datastore_example,
)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize(
    "datastore_boundary_name", DATASTORES_BOUNDARY_EXAMPLES.keys()
)
def test_training(datastore_name, datastore_boundary_name):
    datastore = init_datastore_example(datastore_name)
    datastore_boundary = init_datastore_boundary_example(
        datastore_boundary_name
    )

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )
    if not isinstance(datastore_boundary, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_boundary_name} as it is not a "
            "regular grid datastore."
        )

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        # XXX: `devices` has to be set to 2 otherwise
        # neural_lam.models.ar_model.ARModel.aggregate_and_plot_metrics fails
        # because it expects to aggregate over multiple devices
        devices=2,
        log_every_n_steps=1,
    )

    graph_name = "1level"

    graph_dir_path = Path(datastore.root_path) / "graphs" / graph_name

    def _create_graph():
        if not graph_dir_path.exists():
            build_graph_from_archetype(
                datastore=datastore,
                datastore_boundary=datastore_boundary,
                graph_name=graph_name,
                archetype="keisler",
                mesh_node_distance=get_test_mesh_dist(
                    datastore, datastore_boundary
                ),
            )

    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        ar_steps_train=3,
        ar_steps_eval=5,
        standardize=True,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        # XXX: this should be superfluous when we have already defined the
        # model object no?
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
        num_past_boundary_steps = 1
        num_future_boundary_steps = 1

    model_args = ModelArgs()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(  # noqa
        args=model_args,
        datastore=datastore,
        config=config,
    )
    wandb.init()
    trainer.fit(model=model, datamodule=data_module)
