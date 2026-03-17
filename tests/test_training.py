# Standard library
from pathlib import Path
from types import SimpleNamespace

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


def run_simple_training(
    datastore, set_output_std, loss, devices=2, validate=True
):
    """
    Run one epoch of a simple model training setup using the given datastore.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore to load data from for training
    set_output_std : bool
        If --output_std should be set during training
    loss : str
        Loss function to use during training
    """

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
        devices=devices,
        limit_val_batches=1 if validate else 0,
        num_sanity_val_steps=2 if validate else 0,
        log_every_n_steps=1,
        # use `detect_anomaly` to ensure that we don't have NaNs popping up
        # during training
        detect_anomaly=True,
    )

    graph_name = "1level"

    ensure_graph_exists(datastore, graph_name)

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

    model_args = build_model_args(
        set_output_std=set_output_std,
        loss=loss,
        graph_name=graph_name,
    )

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


def ensure_graph_exists(datastore, graph_name):
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )


def build_model_args(set_output_std, loss, graph_name):
    return SimpleNamespace(
        output_std=set_output_std,
        loss=loss,
        restore_opt=False,
        n_example_pred=1,
        # XXX: this should be superfluous when we have already defined the
        # model object no?
        graph=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=2,
        mesh_aggr="sum",
        lr=1.0e-3,
        val_steps_to_log=[1, 3],
        metrics_watch=[],
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    run_simple_training(datastore, set_output_std=False, loss="mse")


def test_training_output_std():
    datastore = init_datastore_example("dummydata")
    run_simple_training(
        datastore,
        set_output_std=True,
        loss="nll",
        devices=1,
        validate=False,
    )


def test_model_rejects_output_std_with_incompatible_loss():
    datastore = init_datastore_example("dummydata")
    graph_name = "1level"
    ensure_graph_exists(datastore, graph_name)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    with pytest.raises(ValueError, match="--output_std requires a loss"):
        GraphLAM(
            args=build_model_args(
                set_output_std=True, loss="mse", graph_name=graph_name
            ),
            datastore=datastore,
            config=config,
        )
