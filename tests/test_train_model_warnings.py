# Standard library
import inspect
from argparse import Namespace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

# Third-party
import loguru
import pytest


# Mock loguru.logger.catch before importing train_model
def _mock_catch(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f


loguru.logger.catch = _mock_catch  # type: ignore[assignment]

# First-party
from neural_lam.models import MODELS, BaseHiGraphModel  # noqa: E402
from neural_lam.train_model import (  # noqa: E402
    _checkpoint_loading_kwargs,
    build_predictor,
    load_forecasting_module_from_checkpoint,
    main,
)

if TYPE_CHECKING:
    # First-party
    from neural_lam.config import NeuralLAMConfig  # noqa: E402


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({}, {}),
        ({"weights_only": object()}, {"weights_only": False}),
    ],
)
def test_checkpoint_loading_kwargs_follow_lightning_signature(
    parameters, expected
):
    """Pass ``weights_only`` only when Lightning exposes the argument."""
    signature = SimpleNamespace(parameters=parameters)
    with patch(
        "neural_lam.train_model.inspect.signature",
        return_value=signature,
    ):
        assert _checkpoint_loading_kwargs() == expected


@pytest.mark.parametrize(
    "eval_val,load_val,expect_warning",
    [
        ("val", None, True),
        ("val", "path/to/checkpt", False),
        (None, None, False),
    ],
)
def test_eval_without_load_warning(eval_val, load_val, expect_warning):
    mock_args = MagicMock()
    mock_args.eval = eval_val
    mock_args.load = load_val
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = []
    mock_args.train_steps_to_log = []
    mock_args.metrics_watch = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10
    mock_args.restore_opt = False
    mock_args.pretrained_checkpoint = None
    mock_args.pretrained_components = []
    mock_args.freeze_components = []
    mock_args.allow_missing_compatibility_metadata = False

    with patch(
        "neural_lam.train_model.ArgumentParser.parse_args",
        return_value=mock_args,
    ):
        with patch(
            "neural_lam.train_model.load_config_and_datastore",
            side_effect=SystemExit(0),
        ):
            with patch("neural_lam.train_model.logger.warning") as mock_warning:
                with pytest.raises(SystemExit):
                    main()
                if expect_warning:
                    mock_warning.assert_called_once()
                    assert "--load" in mock_warning.call_args[0][0]
                else:
                    mock_warning.assert_not_called()


def test_create_gif_forwarded_to_forecasting_module():
    """--create_gif must be forwarded to
    DeterministicForecastingModule.__init__."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = [1]
    mock_args.train_steps_to_log = [2]
    mock_args.metrics_watch = []
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10
    mock_args.create_gif = True
    mock_args.devices = ["auto"]
    mock_args.model = "graph_lam"
    mock_args.restore_opt = False
    mock_args.pretrained_checkpoint = None
    mock_args.pretrained_components = []
    mock_args.freeze_components = []
    mock_args.allow_missing_compatibility_metadata = False

    captured_kwargs = {}

    def capture_init(_self, **kwargs):
        captured_kwargs.update(kwargs)
        # Raise so we don't need to mock trainer.fit
        raise SystemExit(0)

    with (
        patch(
            "neural_lam.train_model.ArgumentParser.parse_args",
            return_value=mock_args,
        ),
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock(), None),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock}),
        patch("neural_lam.train_model.DeterministicARForecaster"),
        patch(
            "neural_lam.models.modules.deterministic."
            "DeterministicForecastingModule.__init__",
            capture_init,
        ),
        pytest.raises(SystemExit),
    ):
        main()

    assert "create_gif" in captured_kwargs, (
        "create_gif was not forwarded to DeterministicForecastingModule"
    )
    assert captured_kwargs["create_gif"] is True
    assert "train_steps_to_log" in captured_kwargs, (
        "train_steps_to_log was not forwarded to DeterministicForecastingModule"
    )
    assert captured_kwargs["train_steps_to_log"] == [2]


@pytest.mark.parametrize(
    "cli_args,match_err",
    [
        (
            [
                "--train_steps_to_log",
                "1",
                "10",
                "--val_steps_to_log",
                "1",
                "10",
                "--metrics_watch",
                "val_rmse",
                "--var_leads_metrics_watch",
                '{"0": [1, 10]}',
            ],
            None,
        ),
        (["--train_steps_to_log", "15"], "Can not log train step 15"),
        (["--train_steps_to_log", "0"], "Can not log train step 0"),
        (["--val_steps_to_log", "15"], "Can not log val step 15"),
        (["--val_steps_to_log", "0"], "Can not log val step 0"),
        (
            ["--var_leads_metrics_watch", '{"0": [15]}'],
            "Can not log validation step 15",
        ),
        (
            ["--var_leads_metrics_watch", '{"0": [0]}'],
            "Can not log validation step 0",
        ),
        (
            ["--metrics_watch", "val_rmse"],
            "but var_leads_metrics_watch is not set",
        ),
        (
            ["--var_leads_metrics_watch", '{"0": [1]}'],
            "but metrics_watch is not set",
        ),
    ],
)
def test_steps_to_log_validation(cli_args, match_err):
    """ValueError must be raised if a step is outside [1, rollout length]
    or only one of metrics_watch and var_leads_metrics_watch is set."""
    if match_err is None:
        # The checks run before load_config_and_datastore, so hitting the
        # SystemExit sentinel means every step was accepted
        expected = pytest.raises(SystemExit)
    else:
        expected = pytest.raises(ValueError, match=match_err)

    with patch(
        "neural_lam.train_model.load_config_and_datastore",
        side_effect=SystemExit(0),
    ):
        with expected:
            getattr(main, "__wrapped__", main)(
                ["--config_path", "dummy.yaml", "--ar_steps_train", "10"]
                + cli_args
            )


@pytest.mark.parametrize(
    ("cli_args", "match"),
    [
        (["--pretrained_checkpoint", ""], "must not be empty"),
        (
            ["--pretrained_checkpoint", "source.ckpt"],
            "must be set together",
        ),
        (
            ["--pretrained_components", "forecaster.predictor.processor"],
            "must be set together",
        ),
        (
            ["--allow_missing_compatibility_metadata"],
            "requires pretrained_checkpoint",
        ),
        (
            [
                "--load",
                "resume.ckpt",
                "--pretrained_checkpoint",
                "source.ckpt",
                "--pretrained_components",
                "forecaster.predictor.processor",
            ],
            "cannot be combined with load",
        ),
        (
            [
                "--eval",
                "test",
                "--pretrained_checkpoint",
                "source.ckpt",
                "--pretrained_components",
                "forecaster.predictor.processor",
            ],
            "only supported for training",
        ),
        (
            [
                "--restore_opt",
                "--pretrained_checkpoint",
                "source.ckpt",
                "--pretrained_components",
                "forecaster.predictor.processor",
            ],
            "restore_opt cannot be used",
        ),
        (
            [
                "--eval",
                "test",
                "--freeze_components",
                "forecaster.predictor.processor",
            ],
            "freeze_components is only supported for training",
        ),
    ],
)
def test_pretrained_option_validation(cli_args, match):
    """Reject ambiguous selective-loading run configurations early."""
    with pytest.raises(ValueError, match=match):
        getattr(main, "__wrapped__", main)(
            ["--config_path", "dummy.yaml"] + cli_args
        )


def make_args(**overrides):
    """Predictor args as stored in a checkpoint, with per-test overrides."""
    args = dict(
        model="hi_lam",
        graph="hierarchical",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
        loss="wmse",
    )
    args.update(overrides)
    return Namespace(**args)


def capturing_predictor(base=object):
    """Predictor class recording constructor kwargs instead of building."""
    captured_kwargs = {}

    class DummyPredictor(base):
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    return DummyPredictor, captured_kwargs


@pytest.fixture
def config():
    """Config stub exposing only what build_predictor reads."""
    return cast(
        "NeuralLAMConfig",
        SimpleNamespace(
            training=SimpleNamespace(
                output_clamping=SimpleNamespace(lower={}, upper={})
            )
        ),
    )


def test_pretrained_checkpoint_uses_lightning_loader(config):
    """Initialize through Lightning but start Trainer without resume state."""

    class TrainablePredictor:
        """Minimal trainable predictor accepted by the fit setup."""

        trainable = True

        def __init__(self, **_kwargs):
            pass

    datastore = MagicMock()
    datastore.get_vars_names.return_value = []
    component = "forecaster.predictor.processor"
    loaded_model = MagicMock()
    loaded_model.parameters.return_value = [SimpleNamespace(requires_grad=True)]
    trainer = MagicMock(global_rank=0)
    val_checkpoint = MagicMock(best_model_path="")
    latest_checkpoint = MagicMock(best_model_path="")

    with (
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(config, datastore, None),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch(
            "neural_lam.train_model.MODELS",
            {"graph_lam": TrainablePredictor},
        ),
        patch("neural_lam.train_model.DeterministicARForecaster"),
        patch(
            "neural_lam.train_model.DeterministicForecastingModule."
            "load_from_checkpoint",
            return_value=loaded_model,
        ) as load_from_checkpoint,
        patch(
            "neural_lam.train_model.pl.callbacks.ModelCheckpoint",
            side_effect=[val_checkpoint, latest_checkpoint],
        ),
        patch("neural_lam.train_model.pl.Trainer", return_value=trainer),
        patch("neural_lam.train_model.utils.setup_training_logger"),
        patch("neural_lam.train_model.utils.init_training_logger_metrics"),
    ):
        getattr(main, "__wrapped__", main)(
            [
                "--config_path",
                "dummy.yaml",
                "--pretrained_checkpoint",
                "source.ckpt",
                "--pretrained_components",
                component,
                "--freeze_components",
                component,
                "--allow_missing_compatibility_metadata",
            ]
        )

    load_from_checkpoint.assert_called_once()
    call_kwargs = load_from_checkpoint.call_args.kwargs
    assert call_kwargs["pretrained_components"] == [component]
    assert call_kwargs["strict"] is True
    assert call_kwargs["map_location"] == "cpu"
    assert call_kwargs["allow_missing_compatibility_metadata"] is True
    loaded_model.freeze_components.assert_called_once_with([component])
    trainer.fit.assert_called_once()
    assert trainer.fit.call_args.kwargs["model"] is loaded_model
    assert trainer.fit.call_args.kwargs["ckpt_path"] is None


def test_load_keeps_standard_lightning_resume_path(config):
    """Keep normal resume delegated to ``Trainer.fit(ckpt_path=...)``."""

    class TrainablePredictor:
        """Minimal trainable predictor accepted by the fit setup."""

        trainable = True

        def __init__(self, **_kwargs):
            pass

    datastore = MagicMock()
    datastore.get_vars_names.return_value = []
    model = MagicMock()
    trainer = MagicMock(global_rank=0)
    val_checkpoint = MagicMock(best_model_path="")
    latest_checkpoint = MagicMock(best_model_path="")

    with (
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(config, datastore, None),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch(
            "neural_lam.train_model.MODELS",
            {"graph_lam": TrainablePredictor},
        ),
        patch("neural_lam.train_model.DeterministicARForecaster"),
        patch(
            "neural_lam.train_model.DeterministicForecastingModule"
        ) as module_class,
        patch(
            "neural_lam.train_model.pl.callbacks.ModelCheckpoint",
            side_effect=[val_checkpoint, latest_checkpoint],
        ),
        patch("neural_lam.train_model.pl.Trainer", return_value=trainer),
        patch("neural_lam.train_model.utils.setup_training_logger"),
        patch("neural_lam.train_model.utils.init_training_logger_metrics"),
    ):
        module_class.return_value = model
        getattr(main, "__wrapped__", main)(
            [
                "--config_path",
                "dummy.yaml",
                "--load",
                "transferred.ckpt",
            ]
        )

    module_class.load_from_checkpoint.assert_not_called()
    trainer.fit.assert_called_once()
    assert trainer.fit.call_args.kwargs["model"] is model
    assert trainer.fit.call_args.kwargs["ckpt_path"] == "transferred.ckpt"


def test_checkpoint_loader_restores_gnn_type_kwargs(config):
    """Checkpoint reload must preserve custom GNN choices from saved args."""
    args = make_args(
        g2m_gnn_type="PropagationNet",
        m2g_gnn_type="PropagationNet",
        mesh_up_gnn_type="PropagationNet",
        mesh_down_gnn_type="InteractionNet",
    )
    predictor_class, captured_kwargs = capturing_predictor(BaseHiGraphModel)
    loaded_module = MagicMock()

    with (
        patch(
            "neural_lam.train_model.torch.load",
            return_value={"hyper_parameters": {"args": args}},
        ),
        patch("neural_lam.train_model.MODELS", {"hi_lam": predictor_class}),
        patch("neural_lam.train_model.DeterministicARForecaster"),
        patch(
            "neural_lam.train_model.DeterministicForecastingModule"
            ".load_from_checkpoint",
            return_value=loaded_module,
        ),
    ):
        result = load_forecasting_module_from_checkpoint(
            "model.ckpt", config, MagicMock()
        )

    assert result is loaded_module
    assert captured_kwargs["g2m_gnn_type"] == "PropagationNet"
    assert captured_kwargs["m2g_gnn_type"] == "PropagationNet"
    assert captured_kwargs["mesh_up_gnn_type"] == "PropagationNet"
    assert captured_kwargs["mesh_down_gnn_type"] == "InteractionNet"


def test_build_predictor_omits_hierarchical_gnn_kwargs_for_graph_lam(config):
    """GraphLAM must not receive hierarchical-only GNN constructor kwargs."""
    args = make_args(
        model="graph_lam",
        graph="multiscale",
        g2m_gnn_type="PropagationNet",
        m2g_gnn_type="InteractionNet",
        mesh_up_gnn_type="PropagationNet",
        mesh_down_gnn_type="PropagationNet",
    )
    predictor_class, captured_kwargs = capturing_predictor()

    build_predictor(predictor_class, args, config, MagicMock())

    assert "mesh_up_gnn_type" not in captured_kwargs
    assert "mesh_down_gnn_type" not in captured_kwargs
    assert captured_kwargs["g2m_gnn_type"] == "PropagationNet"
    # The dummy swallows **kwargs, so also check against the real signature
    graph_lam_params = inspect.signature(
        MODELS["graph_lam"].__init__
    ).parameters
    assert set(captured_kwargs) <= set(graph_lam_params)


def test_build_predictor_defaults_gnn_types_for_old_checkpoints(config):
    """Checkpoints without GNN type flags fall back to InteractionNet."""
    predictor_class, captured_kwargs = capturing_predictor(BaseHiGraphModel)

    build_predictor(predictor_class, make_args(), config, MagicMock())

    assert captured_kwargs["g2m_gnn_type"] == "InteractionNet"
    assert captured_kwargs["m2g_gnn_type"] == "InteractionNet"
    assert captured_kwargs["mesh_up_gnn_type"] == "InteractionNet"
    assert captured_kwargs["mesh_down_gnn_type"] == "InteractionNet"


def test_build_predictor_adds_hierarchical_kwargs_for_hi_subclass(config):
    """Future BaseHiGraphModel subclasses get hierarchical GNN kwargs."""
    args = make_args(
        model="future_hi_model",
        g2m_gnn_type="InteractionNet",
        m2g_gnn_type="InteractionNet",
        mesh_up_gnn_type="PropagationNet",
        mesh_down_gnn_type="PropagationNet",
    )
    predictor_class, captured_kwargs = capturing_predictor(BaseHiGraphModel)

    build_predictor(predictor_class, args, config, MagicMock())

    assert captured_kwargs["mesh_up_gnn_type"] == "PropagationNet"
    assert captured_kwargs["mesh_down_gnn_type"] == "PropagationNet"
