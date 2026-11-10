"""Tests for the high-level neural_lam Python API."""

# Standard library
import dataclasses

# Third-party
import pytest
import wandb

# First-party
from neural_lam import api, train_model
from neural_lam import config as nlconfig
from neural_lam.models import DeterministicForecastingModule
from neural_lam.train_model import load_forecasting_module_from_checkpoint
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
    reloaded_module = load_forecasting_module_from_checkpoint(
        ckpt_path=str(train_run.checkpoint_path),
        config=config,
        datastore=datastore,
    )
    assert isinstance(reloaded_module, DeterministicForecastingModule)

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
