"""Tests for the high-level neural_lam Python API."""

# Standard library
import dataclasses

# Third-party
import numpy as np
import pytest
import wandb
import xarray as xr

# First-party
from neural_lam import api, train_model
from neural_lam import config as nlconfig
from neural_lam.models import ForecasterModule
from neural_lam.train_model import load_forecaster_module_from_checkpoint
from tests.dummy_datastore import DummyDatastore


@pytest.fixture(scope="module")
def shared_env():
    """Module-scoped datastore and graphs fixture shared by API tests."""
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )
    api.create_graph(
        datastore=datastore,
        name="1level",
        archetype="keisler",
    )
    api.create_graph(
        datastore=datastore,
        name="hierarchical",
        archetype="hierarchical",
        level_refinement_factor=2,
        mesh_grid_distance_ratio=1.5,
    )
    return datastore, config


@pytest.mark.parametrize(
    ("model", "graph"),
    [
        ("graph_lam", "1level"),
        ("hi_lam", "hierarchical"),
        ("hi_lam_parallel", "hierarchical"),
    ],
)
def test_api_train_and_evaluate_roundtrip(shared_env, tmp_path, model, graph):
    """Test train, checkpoint reload, and eval roundtrip with drift gates."""
    datastore, config = shared_env
    runs_root = tmp_path / "runs"

    # Train
    train_run = api.train(
        datastore=datastore,
        config=config,
        model=model,
        graph=graph,
        epochs=1,
        ar_steps_train=1,
        val_interval=1,
        num_sanity_val_steps=0,
        batch_size=1,
        num_workers=0,
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        runs_root=str(runs_root),
        logger_run_name=f"train_{model}",
        val_steps_to_log=[1],
    )

    assert train_run.run_dir.exists()
    # fit closes its W&B run so the following evaluate starts a new one
    assert wandb.run is None
    assert train_run.checkpoint_dir.exists()
    assert train_run.checkpoint_path is not None
    assert train_run.checkpoint_path.exists()

    # Verify reload from checkpoint
    reloaded_module = load_forecaster_module_from_checkpoint(
        ckpt_path=str(train_run.checkpoint_path),
        config=config,
        datastore=datastore,
    )
    assert isinstance(reloaded_module, ForecasterModule)

    # Evaluate with create_gif=True to act as drift gate for all Run paths
    eval_run = api.evaluate(
        datastore=datastore,
        config=config,
        model=model,
        graph=graph,
        load=str(train_run.checkpoint_path),
        batch_size=1,
        num_workers=0,
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        ar_steps_eval=2,
        val_steps_to_log=[1],
        create_gif=True,
        runs_root=str(runs_root),
        logger_run_name=f"eval_{model}",
    )

    # Assert every Run path exists (drift gate)
    assert eval_run.run_dir.exists()
    assert eval_run.checkpoint_path is not None
    assert eval_run.checkpoint_path.exists()
    assert eval_run.plot_dir.exists()
    assert eval_run.rmse_plot.exists()
    assert eval_run.example_plots(1).exists()


def test_api_evaluate_persistence(shared_env, tmp_path):
    """Test evaluate with non-trainable baseline model persistence."""
    datastore, config = shared_env
    eval_run = api.evaluate(
        datastore=datastore,
        config=config,
        model="persistence",
        graph="1level",
        batch_size=1,
        num_workers=0,
        ar_steps_eval=2,
        val_steps_to_log=[1],
        create_gif=True,
        runs_root=str(tmp_path / "runs"),
        logger_run_name="eval_persistence",
    )
    assert eval_run.run_dir.exists()
    assert eval_run.rmse_plot.exists()
    assert eval_run.example_plots(1).exists()


def test_config_parity(monkeypatch, tmp_path):
    """Verify CLI parser dests, api.train(), and dataclasses have parity."""
    # 1. build_parser() dests equal dataclass fields (+ config_path)
    parser = train_model.build_parser()
    parser_dests = {
        action.dest for action in parser._actions if action.dest != "help"
    }
    dataclass_classes = (
        nlconfig.ModelConfig,
        nlconfig.TrainRunConfig,
        nlconfig.DataConfig,
        nlconfig.ComputeConfig,
        nlconfig.LoggingConfig,
    )
    dataclass_fields = {
        f.name for cls in dataclass_classes for f in dataclasses.fields(cls)
    }
    assert parser_dests == dataclass_fields | {"config_path"}

    # 2. Config equality between run(), api.train(), and plain dataclasses
    captured = []

    def mock_fit(
        model_config,
        train_config,
        data_config,
        compute_config,
        logging_config,
        **kwargs,
    ):
        captured.append(
            (
                model_config,
                train_config,
                data_config,
                compute_config,
                logging_config,
            )
        )
        return train_model.Run(run_dir=tmp_path)

    monkeypatch.setattr(train_model, "fit", mock_fit)
    monkeypatch.setattr(api.train_model_script, "fit", mock_fit)

    args = parser.parse_args(["--config_path", "dummy.yaml"])
    train_model.run(args)
    api.train()

    run_cfgs = captured[0]
    api_cfgs = captured[1]
    plain_cfgs = tuple(cls() for cls in dataclass_classes)

    for r, a, p in zip(run_cfgs, api_cfgs, plain_cfgs):
        assert r == p
        assert a == p


def test_fit_config_datastore_validation():
    """Verify config and datastore must be provided together or via path."""
    ds = DummyDatastore(n_grid_points=25, n_timesteps=5)
    cfg = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=ds.SHORT_NAME, config_path=str(ds.root_path)
            )
        }
    )
    model_config = nlconfig.ModelConfig()
    train_config = nlconfig.TrainRunConfig()
    data_config = nlconfig.DataConfig()
    compute_config = nlconfig.ComputeConfig()
    logging_config = nlconfig.LoggingConfig()

    with pytest.raises(
        ValueError, match="config and datastore must be provided together"
    ):
        train_model.fit(
            model_config=model_config,
            train_config=train_config,
            data_config=data_config,
            compute_config=compute_config,
            logging_config=logging_config,
            config=cfg,
            datastore=None,
        )

    with pytest.raises(
        ValueError, match=r"Either \(config and datastore\) or config_path"
    ):
        train_model.fit(
            model_config=model_config,
            train_config=train_config,
            data_config=data_config,
            compute_config=compute_config,
            logging_config=logging_config,
            config=None,
            datastore=None,
            config_path=None,
        )


def test_api_var_leads_metrics_watch_parsing(monkeypatch, tmp_path):
    """Test that var_leads_metrics_watch handles dict, JSON string, and None."""
    captured_train_config = []

    def mock_fit(*, train_config, **kwargs):
        captured_train_config.append(train_config)
        return train_model.Run(run_dir=tmp_path)

    monkeypatch.setattr(api.train_model_script, "fit", mock_fit)

    # Test dict input
    api.train(var_leads_metrics_watch={0: [1, 2]})
    assert captured_train_config[-1].var_leads_metrics_watch == {0: [1, 2]}

    # Test JSON string input
    api.train(var_leads_metrics_watch='{"0": [1, 2]}')
    assert captured_train_config[-1].var_leads_metrics_watch == {0: [1, 2]}

    # Test None input
    api.train(var_leads_metrics_watch=None)
    assert captured_train_config[-1].var_leads_metrics_watch == {}

    # Test evaluate with dict input
    api.evaluate(var_leads_metrics_watch={1: [3, 4]})
    assert captured_train_config[-1].var_leads_metrics_watch == {1: [3, 4]}


def test_create_forecast_dataset_unit(shared_env):
    """
    Test unit behavior of create_forecast_dataset.

    Validates coordinates, dimensions, and ensemble support.
    """
    datastore, _ = shared_env
    b_dim, t_dim, n_dim, f_dim = 2, 3, 100, 5
    preds = np.random.randn(b_dim, t_dim, n_dim, f_dim).astype(np.float32)
    analysis_times = [
        np.datetime64("2021-01-01T02:00:00", "ns"),
        np.datetime64("2021-01-01T03:00:00", "ns"),
    ]
    elapsed = [
        np.timedelta64(1, "h"),
        np.timedelta64(2, "h"),
        np.timedelta64(3, "h"),
    ]

    ds = api.create_forecast_dataset(
        predictions=preds,
        analysis_times=analysis_times,
        elapsed_durations=elapsed,
        datastore=datastore,
    )
    assert isinstance(ds, xr.Dataset)
    assert "state" in ds
    assert ds.state.dims == (
        "analysis_time",
        "elapsed_forecast_duration",
        "grid_index",
        "state_feature",
    )
    assert ds.state.shape == (2, 3, 100, 5)
    assert "valid_time" in ds.coords
    assert ds.coords["valid_time"].shape == (2, 3)
    expected_valid_0_1 = np.datetime64(
        "2021-01-01T02:00:00", "ns"
    ) + np.timedelta64(2, "h")
    assert ds.coords["valid_time"].values[0, 1] == expected_valid_0_1
    assert "x" in ds.coords
    assert "y" in ds.coords

    # Test with ensemble member dimension
    preds_ens = np.random.randn(b_dim, t_dim, 4, n_dim, f_dim).astype(
        np.float32
    )
    ds_ens = api.create_forecast_dataset(
        predictions=preds_ens,
        analysis_times=analysis_times,
        elapsed_durations=elapsed,
        datastore=datastore,
        ensemble_members=["m0", "m1", "m2", "m3"],
    )
    assert ds_ens.state.dims == (
        "analysis_time",
        "elapsed_forecast_duration",
        "ensemble_member",
        "grid_index",
        "state_feature",
    )
    assert list(ds_ens.ensemble_member.values) == ["m0", "m1", "m2", "m3"]


@pytest.mark.requires_training
def test_api_predict_and_init_times(shared_env, tmp_path):
    """
    Test api.predict end-to-end.

    Validates unstandardization, target return, unstack_grid, and init_times.
    """
    datastore, config = shared_env
    runs_root = tmp_path / "runs_predict"

    train_run = api.train(
        datastore=datastore,
        config=config,
        model="graph_lam",
        graph="1level",
        epochs=1,
        ar_steps_train=1,
        val_interval=1,
        num_sanity_val_steps=0,
        batch_size=1,
        num_workers=0,
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        runs_root=str(runs_root),
        logger_run_name="train_predict_ckpt",
        val_steps_to_log=[1],
    )
    ckpt_path = str(train_run.checkpoint_path)

    # 1. Default prediction: init_times=None, unstandardize=True
    ds_pred = api.predict(
        load=ckpt_path,
        datastore=datastore,
        config=config,
        lead_steps=2,
    )
    assert isinstance(ds_pred, xr.Dataset)
    assert "state" in ds_pred
    assert ds_pred.state.dims == (
        "analysis_time",
        "elapsed_forecast_duration",
        "grid_index",
        "state_feature",
    )
    assert ds_pred.state.shape == (1, 2, 100, 5)
    assert not np.isnan(ds_pred.state.values).any()

    # 2. Prediction with return_target=True and unstandardize=False
    ds_pred_std = api.predict(
        load=ckpt_path,
        datastore=datastore,
        config=config,
        lead_steps=2,
        unstandardize=False,
        return_target=True,
    )
    assert ds_pred_std.state.shape == ds_pred_std.state_target.shape
    stats = datastore.get_standardization_dataarray("state")
    np.testing.assert_allclose(
        (ds_pred_std.state * stats.state_std + stats.state_mean)
        .transpose(*ds_pred.state.dims)
        .values,
        ds_pred.state.values,
        rtol=1e-5,
    )

    # 3. Test unstack_grid=True on regular grid
    ds_unstacked = api.predict(
        load=ckpt_path,
        datastore=datastore,
        config=config,
        lead_steps=2,
        unstack_grid=True,
    )
    assert "x" in ds_unstacked.state.dims
    assert "y" in ds_unstacked.state.dims
    assert ds_unstacked.state.sizes["x"] == 10
    assert ds_unstacked.state.sizes["y"] == 10
    assert ds_unstacked.state.sizes["state_feature"] == 5
    assert ds_unstacked.state.sizes["elapsed_forecast_duration"] == 2
    assert ds_unstacked.state.sizes["analysis_time"] == 1

    # 4. Physical targets match the datastore at valid_time
    da_target = api.predict(
        load=ckpt_path,
        datastore=datastore,
        config=config,
        init_times=slice(1, 3),
        lead_steps=2,
        return_target=True,
    ).state_target
    step = np.timedelta64(datastore.step_length)
    times = da_target.analysis_time.values
    assert times[0] == ds_pred.analysis_time.values[0] + step
    np.testing.assert_array_equal(
        da_target.elapsed_forecast_duration.values, [step, 2 * step]
    )
    da_state = datastore.get_dataarray(category="state", split="test")
    np.testing.assert_allclose(
        da_target.values,
        da_state.sel(time=da_target.valid_time).values,
    )

    # 5. Timestamp list and (inclusive) timestamp slice select the same
    for init_times in ([str(t) for t in times], slice(times[0], times[-1])):
        ds_by_time = api.predict(
            load=ckpt_path,
            datastore=datastore,
            config=config,
            init_times=init_times,
            lead_steps=2,
        )
        np.testing.assert_array_equal(ds_by_time.analysis_time.values, times)

    # 6. Unknown init_times raise
    with pytest.raises(KeyError):
        api.predict(
            load=ckpt_path,
            datastore=datastore,
            config=config,
            init_times="2099-01-01T00:00:00",
            lead_steps=2,
        )
