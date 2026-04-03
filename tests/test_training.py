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
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


def run_simple_training(
    datastore,
    set_output_std,
    loss,
    metrics_watch=None,
    devices=None,
    validate=True,
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
    metrics_watch : list[str] | None
        Optional metrics_watch configuration for evaluation
    devices : int | None
        Number of devices to use in the Lightning trainer. Defaults to a
        single CPU device in CPU-only test environments and two devices when
        CUDA is available, so the test remains cross-platform while still
        exercising the multi-device path on GPU runners.
    validate : bool
        Whether validation should run during the training test
    """

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    if devices is None:
        devices = 2 if device_name == "cuda" else 1

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
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
        metrics_watch=metrics_watch,
    )

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
    wandb.init(mode="disabled")  # Disable wandb for offline test run
    trainer.fit(model=model, datamodule=data_module)


def ensure_graph_exists(datastore, graph_name):
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )


def build_model_args(set_output_std, loss, graph_name, metrics_watch=None):
    watched_metrics = metrics_watch or []
    watched_var_steps = {0: [1]} if watched_metrics else {}
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
        metrics_watch=watched_metrics,
        var_leads_metrics_watch=watched_var_steps,
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


def test_all_gather_cat_single_device():
    """
    Test that all_gather_cat preserves tensor shape on single-device runs.
    On a single device, all_gather returns the tensor unchanged (no new
    leading dim), so all_gather_cat should not flatten any existing dims.
    """

    class MockModule:
        """Minimal object with mocked single-device all_gather."""

        def all_gather(self, tensor_to_gather, sync_grads=False):
            return tensor_to_gather

    module = MockModule()
    module.all_gather_cat = ARModel.all_gather_cat.__get__(module, MockModule)

    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)

    assert result.shape == tensor.shape, (
        f"all_gather_cat changed shape on single device: "
        f"{tensor.shape} -> {result.shape}"
    )
    assert torch.equal(result, tensor)


def test_all_gather_cat_multi_device_simulation():
    """
    Test that all_gather_cat correctly flattens when all_gather adds a
    leading dimension (simulating multi-device behavior).
    """

    class MockModule:
        """Object with mocked multi-device all_gather."""

        def all_gather(self, tensor, sync_grads=False):
            return torch.stack([tensor, tensor], dim=0)

    module = MockModule()
    module.all_gather_cat = ARModel.all_gather_cat.__get__(module, MockModule)

    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)

    assert result.shape == (
        8,
        3,
        5,
    ), f"all_gather_cat wrong shape on multi-device: {result.shape}"
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(result, expected), (
        "all_gather_cat produced incorrectly ordered/combined values "
        "on multi-device simulation"
    )
