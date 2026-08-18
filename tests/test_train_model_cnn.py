# Standard library
from argparse import Namespace
from types import SimpleNamespace

# Third-party
import pytest
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models import (
    MODELS,
    ARForecaster,
    CNNPredictor,
    ForecasterModule,
)
from neural_lam.train_model import (
    build_predictor,
    get_predictor_kwargs,
    load_forecaster_module_from_checkpoint,
    main,
)
from neural_lam.weather_dataset import WeatherDataModule
from tests.dummy_datastore import DummyDatastore


def _config():
    return SimpleNamespace(
        training=SimpleNamespace(
            output_clamping=SimpleNamespace(lower={}, upper={})
        )
    )


def _neural_lam_config():
    return nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp",
            config_path=".",
        )
    )


def _cnn_args(**overrides):
    args = Namespace(
        model="cnn_predictor",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
        cnn_channels=8,
        cnn_blocks=2,
        cnn_kernel_size=3,
        cnn_se_reduction=4,
        cnn_film=False,
        cnn_padding_mode="zeros",
        loss="mse",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        create_gif=False,
        val_steps_to_log=[1],
        metrics_watch=[],
        var_leads_metrics_watch={},
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_cnn_forecaster_module(datastore, config, args):
    predictor = build_predictor(args, config, datastore)
    forecaster = ARForecaster(predictor, datastore)
    return ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss=args.loss,
        lr=args.lr,
        restore_opt=args.restore_opt,
        n_example_pred=args.n_example_pred,
        val_steps_to_log=args.val_steps_to_log,
        metrics_watch=args.metrics_watch,
        var_leads_metrics_watch=args.var_leads_metrics_watch,
        args=args,
    )


def test_cnn_predictor_registered_in_models():
    assert MODELS["cnn_predictor"] is CNNPredictor


def test_cnn_predictor_kwargs_do_not_include_graph_options():
    kwargs = get_predictor_kwargs(_cnn_args(cnn_film=True), _config())

    assert kwargs["cnn_channels"] == 8
    assert kwargs["cnn_blocks"] == 2
    assert kwargs["cnn_film"] is True
    assert "graph_name" not in kwargs
    assert "hidden_dim" not in kwargs
    assert "g2m_gnn_type" not in kwargs


def test_build_predictor_constructs_cnn_predictor():
    datastore = DummyDatastore(n_grid_points=16)
    predictor = build_predictor(
        _cnn_args(cnn_padding_mode="reflect"),
        _config(),
        datastore,
    )

    assert isinstance(predictor, CNNPredictor)
    assert predictor.backbone.blocks[0].conv1.padding_mode == "reflect"


def test_train_model_cli_constructs_cnn_predictor(monkeypatch):
    datastore = DummyDatastore(n_grid_points=16)
    captured = {}

    def capture_forecaster_module_init(_self, **kwargs):
        captured["predictor"] = kwargs["forecaster"].predictor
        raise SystemExit(0)

    monkeypatch.setattr(
        "neural_lam.train_model.load_config_and_datastore",
        lambda config_path: (_config(), datastore),
    )
    monkeypatch.setattr(
        "neural_lam.train_model.WeatherDataModule",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "neural_lam.train_model.ForecasterModule.__init__",
        capture_forecaster_module_init,
    )

    with pytest.raises(SystemExit):
        main(
            [
                "--config_path",
                "dummy.yaml",
                "--model",
                "cnn_predictor",
                "--cnn_channels",
                "6",
                "--cnn_blocks",
                "1",
                "--cnn_se_reduction",
                "2",
                "--cnn_film",
                "--cnn_padding_mode",
                "reflect",
                "--ar_steps_eval",
                "1",
                "--val_steps_to_log",
                "1",
            ]
        )

    predictor = captured["predictor"]
    assert isinstance(predictor, CNNPredictor)
    assert predictor.cnn_film is True
    assert predictor.backbone.blocks[0].conv1.padding_mode == "reflect"


def test_load_cnn_predictor_from_checkpoint(tmp_path):
    datastore = DummyDatastore(n_grid_points=16)
    config = _neural_lam_config()
    args = _cnn_args(cnn_padding_mode="reflect")
    model = _make_cnn_forecaster_module(datastore, config, args)

    ckpt_path = tmp_path / "cnn_predictor.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    loaded_model = load_forecaster_module_from_checkpoint(
        ckpt_path,
        config,
        datastore,
    )

    assert isinstance(loaded_model.forecaster.predictor, CNNPredictor)
    assert (
        loaded_model.forecaster.predictor.backbone.blocks[0].conv1.padding_mode
        == "reflect"
    )

    batch_size = 2
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * 3
    init_states = torch.randn(batch_size, 2, num_grid_nodes, d_state)
    forcing = torch.randn(batch_size, 1, num_grid_nodes, d_forcing)
    boundary = torch.randn(batch_size, 1, num_grid_nodes, d_state)

    with torch.no_grad():
        prediction_before, _ = model.forecaster(init_states, forcing, boundary)
        prediction_after, _ = loaded_model.forecaster(
            init_states,
            forcing,
            boundary,
        )

    assert torch.allclose(prediction_before, prediction_after)


def test_cnn_predictor_training_smoke():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=6)
    config = _neural_lam_config()
    args = _cnn_args(cnn_channels=4, cnn_blocks=1, cnn_se_reduction=2)
    model = _make_cnn_forecaster_module(datastore, config, args)
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=1,
        ar_steps_eval=1,
        batch_size=2,
        num_workers=0,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=data_module)
