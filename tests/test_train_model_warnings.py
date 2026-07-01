# Standard library
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third-party
import pytest

# First-party
from neural_lam.train_model import (
    build_predictor,
    load_forecaster_module_from_checkpoint,
    main,
)


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
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10

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
    mock_args.var_leads_metrics_watch = "{}"
    mock_args.ar_steps_eval = 10
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
        patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock()}),
        patch("neural_lam.train_model.ARForecaster"),
        patch(
            "neural_lam.models.module.ForecasterModule.__init__",
            capture_init,
        ),
        pytest.raises(SystemExit),
    ):
        main()

    assert (
        "create_gif" in captured_kwargs
    ), "create_gif was not forwarded to ForecasterModule"
    assert captured_kwargs["create_gif"] is True


def test_checkpoint_loader_restores_gnn_type_kwargs():
    """Checkpoint reload must preserve custom GNN choices from saved args."""
    args = SimpleNamespace(
        model="hi_lam",
        graph="hierarchical",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
        g2m_gnn_type="PropagationNet",
        m2g_gnn_type="PropagationNet",
        mesh_up_gnn_type="PropagationNet",
        mesh_down_gnn_type="InteractionNet",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            output_clamping=SimpleNamespace(lower={}, upper={})
        )
    )
    datastore = MagicMock()
    captured_kwargs = {}

    class DummyPredictor:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    loaded_module = MagicMock()

    with (
        patch(
            "neural_lam.train_model.torch.load",
            return_value={"hyper_parameters": {"args": args}},
        ),
        patch("neural_lam.train_model.MODELS", {"hi_lam": DummyPredictor}),
        patch("neural_lam.train_model.ARForecaster"),
        patch(
            "neural_lam.train_model.ForecasterModule.load_from_checkpoint",
            return_value=loaded_module,
        ),
    ):
        result = load_forecaster_module_from_checkpoint(
            "model.ckpt", config, datastore
        )

    assert result is loaded_module
    assert captured_kwargs["g2m_gnn_type"] == "PropagationNet"
    assert captured_kwargs["m2g_gnn_type"] == "PropagationNet"
    assert captured_kwargs["mesh_up_gnn_type"] == "PropagationNet"
    assert captured_kwargs["mesh_down_gnn_type"] == "InteractionNet"


def test_build_predictor_omits_hierarchical_gnn_kwargs_for_graph_lam():
    """GraphLAM must not receive hierarchical-only GNN constructor kwargs."""
    args = SimpleNamespace(
        model="graph_lam",
        graph="multiscale",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
        g2m_gnn_type="PropagationNet",
        m2g_gnn_type="InteractionNet",
        mesh_up_gnn_type="PropagationNet",
        mesh_down_gnn_type="PropagationNet",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            output_clamping=SimpleNamespace(lower={}, upper={})
        )
    )
    captured_kwargs = {}

    class DummyGraphLAM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    build_predictor(DummyGraphLAM, args, config, MagicMock())

    assert "mesh_up_gnn_type" not in captured_kwargs
    assert "mesh_down_gnn_type" not in captured_kwargs
    assert captured_kwargs["g2m_gnn_type"] == "PropagationNet"
