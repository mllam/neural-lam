# Standard library
from pathlib import Path

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
from neural_lam.models.forecaster_module import ForecasterModule
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example

# Model architecture defaults for tests
GRAPH = "1level"
HIDDEN_DIM = 4
HIDDEN_LAYERS = 1
PROCESSOR_LAYERS = 2
MESH_AGGR = "sum"
NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def run_simple_training(datastore, set_output_std):
    """
    Run one epoch of a simple model training setup using the given datastore.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore to load data from for training
    set_output_std : bool
        If --output_std should be set during training
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
        devices=2,
        log_every_n_steps=1,
        # use `detect_anomaly` to ensure that we don't have NaNs popping up
        # during training
        detect_anomaly=True,
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

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    # First-party
    from neural_lam.models import MODELS
    from neural_lam.models.ar_forecaster import ARForecaster

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=HIDDEN_DIM,
        hidden_layers=HIDDEN_LAYERS,
        processor_layers=PROCESSOR_LAYERS,
        mesh_aggr=MESH_AGGR,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=set_output_std,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1, 3],
        metrics_watch=[],
        var_leads_metrics_watch={},
    )
    wandb.init()
    try:
        trainer.fit(model=model, datamodule=data_module)
    finally:
        wandb.finish()


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    run_simple_training(datastore, set_output_std=False)


def test_training_output_std():
    datastore = init_datastore_example("mdp")  # Test only with mdp datastore
    run_simple_training(datastore, set_output_std=True)
