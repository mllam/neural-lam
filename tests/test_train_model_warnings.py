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
loguru.logger.catch = lambda f: f  # type: ignore[assignment]

# First-party
from neural_lam.models import MODELS, BaseHiGraphModel  # noqa: E402
from neural_lam.train_model import (  # noqa: E402
    build_predictor,
    load_forecaster_module_from_checkpoint,
    main,
)

if TYPE_CHECKING:
    # First-party
    from neural_lam.config import NeuralLAMConfig  # noqa: E402


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
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10

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


def test_create_gif_forwarded_to_forecaster_module():
    """--create_gif must be forwarded to ForecasterModule.__init__."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = [1]
    mock_args.train_steps_to_log = [2]
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10
    mock_args.create_gif = True
    mock_args.devices = ["auto"]
    mock_args.model = "graph_lam"

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
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock}),
        patch("neural_lam.train_model.ARForecaster"),
        patch(
            "neural_lam.models.module.ForecasterModule.__init__",
            capture_init,
        ),
        pytest.raises(SystemExit),
    ):
        main()

    assert "create_gif" in captured_kwargs, (
        "create_gif was not forwarded to ForecasterModule"
    )
    assert captured_kwargs["create_gif"] is True
    assert "train_steps_to_log" in captured_kwargs, (
        "train_steps_to_log was not forwarded to ForecasterModule"
    )
    assert captured_kwargs["train_steps_to_log"] == [2]


@pytest.mark.parametrize(
    "train_steps,val_steps,match_err",
    [
        ([15], [], "Can not log train step 15"),
        ([], [15], "Can not log val step 15"),
    ],
)
def test_steps_to_log_validation(train_steps, val_steps, match_err):
    """ValueError must be raised if steps exceed the rollout length."""
    mock_args = MagicMock()
    mock_args.eval = None
    mock_args.load = None
    mock_args.config_path = "dummy.yaml"
    mock_args.val_steps_to_log = val_steps
    mock_args.train_steps_to_log = train_steps
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
    mock_args.ar_steps_train = 10

    with patch(
        "neural_lam.train_model.ArgumentParser.parse_args",
        return_value=mock_args,
    ):
        with patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock()),
        ):
            with pytest.raises(ValueError, match=match_err):
                getattr(main, "__wrapped__", main)()


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
        patch("neural_lam.train_model.ARForecaster"),
        patch(
            "neural_lam.train_model.ForecasterModule.load_from_checkpoint",
            return_value=loaded_module,
        ),
    ):
        result = load_forecaster_module_from_checkpoint(
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
