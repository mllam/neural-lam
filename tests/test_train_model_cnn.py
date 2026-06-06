# Standard library
from argparse import Namespace
from types import SimpleNamespace

# Third-party
import pytest

# First-party
from neural_lam.models import MODELS, CNNPredictor
from neural_lam.train_model import build_predictor, get_predictor_kwargs, main
from tests.dummy_datastore import DummyDatastore


def _config():
    return SimpleNamespace(
        training=SimpleNamespace(
            output_clamping=SimpleNamespace(lower={}, upper={})
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
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


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
