# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


class ModelArgs:
    """Minimal args object for testing zarr eval saving."""

    output_std = False
    loss = "mse"
    restore_opt = False
    n_example_pred = 0  # skip plotting to keep the test fast
    graph = "1level"
    hidden_dim = 4
    hidden_layers = 1
    processor_layers = 2
    mesh_aggr = "sum"
    lr = 1.0e-3
    val_steps_to_log = [1, 2]
    metrics_watch = []
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    save_eval_to_zarr_path = None  # overridden per test


def run_zarr_eval(datastore, zarr_path, tmp_path):
    """
    Run one test epoch using the given datastore and save predictions to
    *zarr_path*. Returns the opened ``xr.Dataset``.
    """
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        devices=1,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
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
        ar_steps_eval=3,
        standardize=True,
        batch_size=2,
        num_workers=0,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    model_args = ModelArgs()
    model_args.save_eval_to_zarr_path = str(zarr_path)

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(args=model_args, datastore=datastore, config=config)
    trainer.test(model=model, datamodule=data_module)

    return xr.open_zarr(str(zarr_path))


def test_zarr_eval_single_gpu(tmp_path):
    """
    Single-GPU integration test: run eval on DummyDatastore and assert that
    the Zarr output has the expected structure and correct time semantics.
    """
    datastore = init_datastore_example("dummydata")
    zarr_path = tmp_path / "eval_preds.zarr"

    ds = run_zarr_eval(datastore, zarr_path, tmp_path)

    # 1. Store must exist and contain "state"
    assert zarr_path.exists(), "Zarr store was not created"
    assert "state" in ds.data_vars

    # 2. Required dimensions
    required_dims = {"start_time", "elapsed_forecast_duration"}
    missing = required_dims - set(ds.dims)
    assert not missing, f"Missing dimensions: {missing}"

    # 3. Raw 'time' coord must be absent
    assert "time" not in ds.coords

    # 4. start_time must be analysis_time (= first forecast time - step_length)
    step_ns = int(
        ds["elapsed_forecast_duration"].values[0] / np.timedelta64(1, "ns")
    )
    for t0 in ds["start_time"].values:
        first_fcst_abs = t0 + ds["elapsed_forecast_duration"].values[0]
        expected_t0 = first_fcst_abs - np.timedelta64(step_ns, "ns")
        assert t0 == expected_t0, (
            f"start_time {t0} != expected {expected_t0}"
        )

    # 5. Values must be finite (confirms rescaling happened)
    state_sample = float(
        ds["state"].isel(start_time=0, elapsed_forecast_duration=0).mean()
    )
    assert np.isfinite(state_sample), "Zarr output contains NaN/Inf"
