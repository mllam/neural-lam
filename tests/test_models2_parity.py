# Standard library
from pathlib import Path
from types import SimpleNamespace

# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.metrics import crps_loss
from neural_lam.models.graph_lam import GraphLAM as LegacyGraphLAM
from neural_lam.models2.ar_forecast_sampler import ARForecastSampler
from neural_lam.models2.ar_forecaster import ARForecaster
from neural_lam.models2.ensemble_forecaster_module import (
    EnsembleForecasterModule,
)
from neural_lam.models2.forecaster_module import ForecasterModule
from neural_lam.models2.graph_lam import GraphLAM as GraphLAMV2
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example


def _ensure_graph_exists(datastore, graph_name="1level"):
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )


def _build_config(datastore):
    return nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )


def _build_args(
    *,
    output_std=False,
    ensemble_size=1,
    graph_name="1level",
):
    return SimpleNamespace(
        output_std=output_std,
        loss="mse",
        restore_opt=False,
        n_example_pred=0,
        graph=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=2,
        mesh_aggr="sum",
        lr=1.0e-3,
        val_steps_to_log=[1, 2, 3],
        metrics_watch=[],
        var_leads_metrics_watch={},
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        ensemble_size=ensemble_size,
    )


def _build_batch(datastore, args, split="train", batch_size=2, ar_steps=3):
    dataset = WeatherDataset(
        datastore=datastore,
        split=split,
        ar_steps=ar_steps,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
    )
    samples = [dataset[i] for i in range(batch_size)]
    batch = tuple(torch.stack([sample[i] for sample in samples]) for i in range(4))
    return batch


def _build_legacy_and_v2_models(datastore, args, config):
    legacy = LegacyGraphLAM(args=args, config=config, datastore=datastore)
    model_v2 = GraphLAMV2(args=args, config=config, datastore=datastore)

    # Ensure both predictors have identical weights/buffers for parity checks.
    load_result = model_v2.load_state_dict(legacy.state_dict(), strict=False)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0
    return legacy, model_v2


def test_models2_predict_step_parity_against_legacy_graphlam():
    datastore = init_datastore_example("dummydata")
    args = _build_args(output_std=False)
    config = _build_config(datastore)
    _ensure_graph_exists(datastore, graph_name=args.graph)

    legacy, model_v2 = _build_legacy_and_v2_models(datastore, args, config)
    batch = _build_batch(datastore, args, batch_size=1, ar_steps=2)
    init_states, _, forcing_features, _ = batch

    prev_prev_state = init_states[:, 0]
    prev_state = init_states[:, 1]
    forcing = forcing_features[:, 0]

    with torch.no_grad():
        legacy_pred, legacy_std = legacy.predict_step(
            prev_state, prev_prev_state, forcing
        )
        v2_pred, v2_std = model_v2.predict_step(
            prev_state, prev_prev_state, forcing
        )

    assert torch.allclose(legacy_pred, v2_pred, atol=1e-6, rtol=1e-5)
    if legacy_std is None:
        assert v2_std is None
    else:
        assert torch.allclose(legacy_std, v2_std, atol=1e-6, rtol=1e-5)


def test_models2_ar_forecaster_unroll_parity_against_legacy_unroll():
    datastore = init_datastore_example("dummydata")
    args = _build_args(output_std=False, ensemble_size=1)
    config = _build_config(datastore)
    _ensure_graph_exists(datastore, graph_name=args.graph)

    legacy, model_v2 = _build_legacy_and_v2_models(datastore, args, config)
    init_states, target_states, forcing_features, _ = _build_batch(
        datastore, args, batch_size=2, ar_steps=3
    )

    with torch.no_grad():
        legacy_prediction, legacy_pred_std = legacy.unroll_prediction(
            init_states, forcing_features, target_states
        )
        forecaster = ARForecaster(step_predictor=model_v2, args=args)
        v2_prediction, v2_pred_std = forecaster(
            init_states=init_states,
            forcing_features=forcing_features,
            true_states=target_states,
        )

    assert torch.allclose(
        legacy_prediction, v2_prediction, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(legacy_pred_std, v2_pred_std, atol=1e-6, rtol=1e-5)


def test_models2_forecaster_module_common_step_parity_against_legacy():
    datastore = init_datastore_example("dummydata")
    args = _build_args(output_std=False, ensemble_size=1)
    config = _build_config(datastore)
    _ensure_graph_exists(datastore, graph_name=args.graph)

    legacy, model_v2 = _build_legacy_and_v2_models(datastore, args, config)
    batch = _build_batch(datastore, args, batch_size=2, ar_steps=3)

    with torch.no_grad():
        legacy_prediction, legacy_target, legacy_std, legacy_times = (
            legacy.common_step(batch)
        )

        forecaster = ARForecaster(step_predictor=model_v2, args=args)
        module_v2 = ForecasterModule(
            forecaster=forecaster,
            args=args,
            config=config,
            datastore=datastore,
        )
        v2_prediction, v2_target, v2_std, v2_times = module_v2.common_step(batch)

    assert torch.allclose(
        legacy_prediction, v2_prediction, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(legacy_target, v2_target, atol=1e-6, rtol=1e-5)
    assert torch.allclose(legacy_std, v2_std, atol=1e-6, rtol=1e-5)
    assert torch.equal(legacy_times, v2_times)


def test_ar_forecast_sampler_zero_noise_matches_repeated_deterministic_rollout():
    datastore = init_datastore_example("dummydata")
    args = _build_args(output_std=False, ensemble_size=3)
    config = _build_config(datastore)
    _ensure_graph_exists(datastore, graph_name=args.graph)

    _, model_v2 = _build_legacy_and_v2_models(datastore, args, config)
    init_states, target_states, forcing_features, _ = _build_batch(
        datastore, args, batch_size=2, ar_steps=3
    )

    deterministic_forecaster = ARForecaster(step_predictor=model_v2, args=args)
    sampler = ARForecastSampler(
        step_predictor=model_v2,
        args=args,
        noise_std=0.0,
    )

    with torch.no_grad():
        deterministic_prediction, deterministic_std = deterministic_forecaster(
            init_states=init_states,
            forcing_features=forcing_features,
            true_states=target_states,
        )
        sampled_prediction, sampled_std = sampler(
            init_states=init_states,
            forcing_features=forcing_features,
            true_states=target_states,
            ensemble_size=args.ensemble_size,
        )

    assert sampled_prediction.shape[:2] == (
        init_states.shape[0],
        args.ensemble_size,
    )
    expected_prediction = deterministic_prediction.unsqueeze(1).expand_as(
        sampled_prediction
    )
    assert torch.allclose(
        sampled_prediction, expected_prediction, atol=1e-6, rtol=1e-5
    )
    # For deterministic predictor, std is per-variable tensor.
    assert torch.allclose(sampled_std, deterministic_std, atol=1e-6, rtol=1e-5)


def test_ensemble_forecaster_module_training_and_validation_metrics():
    datastore = init_datastore_example("dummydata")
    args = _build_args(output_std=False, ensemble_size=4)
    config = _build_config(datastore)
    _ensure_graph_exists(datastore, graph_name=args.graph)

    _, model_v2 = _build_legacy_and_v2_models(datastore, args, config)
    sampler = ARForecastSampler(
        step_predictor=model_v2,
        args=args,
        noise_std=0.0,
    )
    module = EnsembleForecasterModule(
        forecaster=sampler,
        args=args,
        config=config,
        datastore=datastore,
    )
    batch = _build_batch(datastore, args, batch_size=2, ar_steps=3)

    train_logs = {}
    val_logs = {}

    def _fake_log(name, value, **kwargs):
        train_logs[name] = value.detach() if torch.is_tensor(value) else value
        train_logs[f"{name}_kwargs"] = kwargs

    def _fake_log_dict(log_dict, **kwargs):
        for key, value in log_dict.items():
            val_logs[key] = value.detach() if torch.is_tensor(value) else value
        val_logs["kwargs"] = kwargs

    module.log = _fake_log
    module.log_dict = _fake_log_dict

    train_loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(train_loss)
    assert "train_loss" in train_logs
    assert torch.isfinite(train_logs["train_loss"])

    # Verify training CRPS computation matches explicit computation.
    prediction, target_states, _, _ = module.common_step(batch)
    mask = module.forecaster.step_predictor.interior_mask
    expected_crps = torch.mean(
        crps_loss(
            prediction,
            target_states,
            mask=mask,
            average_grid=False,
            sum_vars=False,
        )
    )
    assert torch.allclose(train_loss, expected_crps, atol=1e-6, rtol=1e-5)

    module.validation_step(batch, batch_idx=0)
    assert "val_crps" in val_logs
    assert "val_spread" in val_logs
    assert "val_error" in val_logs
    assert torch.isfinite(val_logs["val_crps"])
    assert torch.isfinite(val_logs["val_spread"])
    assert torch.isfinite(val_logs["val_error"])
