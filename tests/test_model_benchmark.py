"""Tests for the forecast benchmark evaluation suite."""

# Standard library
from pathlib import Path

# Third-party
import pytest

# First-party
from neural_lam import config as nlconfig
from neural_lam.benchmark import BenchmarkScorecard, ForecastBenchmark
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


def test_forecast_benchmark_with_dummy_datastore():
    """Verify ForecastBenchmark runs fast evaluation on DummyDatastore."""
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=10)
    graph_name = "1level_test_bench"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )

    predictor = GraphLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = ARForecaster(predictor, datastore)
    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
    )

    eval_steps = 2
    benchmark = ForecastBenchmark(
        datastore=datastore,
        split="test",
        eval_steps=eval_steps,
        batch_size=1,
        buffer_width=0,
        fss_kernel_sizes=[1, 3],
        device="cpu",
    )

    scorecard = benchmark.evaluate(model=model, max_batches=2)

    assert isinstance(scorecard, BenchmarkScorecard)
    assert scorecard.is_stable is True
    assert scorecard.num_rollout_steps == eval_steps
    assert scorecard.model_name == "GraphLAM"
    assert len(scorecard.variables) == len(datastore.get_vars_names("state"))

    for var in scorecard.variables:
        rmse_lead = scorecard.rmse_per_lead_time[var]
        bias_lead = scorecard.mbe_per_lead_time[var]
        stde_lead = scorecard.stde_per_lead_time[var]
        assert len(rmse_lead) == eval_steps
        assert len(bias_lead) == eval_steps
        assert len(stde_lead) == eval_steps
        assert all(isinstance(v, float) and v >= 0.0 for v in rmse_lead)
        assert all(isinstance(v, float) for v in bias_lead)

    # Test verdict logic
    passed, msgs = scorecard.evaluate_verdict(mode="operational")
    assert passed is True
    assert any("PASS" in m for m in msgs)

    # Compare with reference scorecard
    ref_scorecard = scorecard
    passed_ref, _ = scorecard.evaluate_verdict(
        reference_scorecard=ref_scorecard, mode="operational"
    )
    assert passed_ref is True

    # Test unstable model verdict failure
    scorecard_unstable = BenchmarkScorecard(
        model_name="UnstableModel",
        num_rollout_steps=eval_steps,
        variables=scorecard.variables,
        lead_time_hours=scorecard.lead_time_hours,
        mse_per_lead_time=scorecard.mse_per_lead_time,
        rmse_per_lead_time=scorecard.rmse_per_lead_time,
        mbe_per_lead_time=scorecard.mbe_per_lead_time,
        stde_per_lead_time=scorecard.stde_per_lead_time,
        is_stable=False,
    )
    passed_unstable, msgs_unstable = scorecard_unstable.evaluate_verdict(
        mode="operational"
    )
    assert passed_unstable is False
    assert any("FAIL" in m for m in msgs_unstable)

    summary_text = scorecard.summary()
    assert "Forecast Benchmark Summary" in summary_text
    assert "RMSE" in summary_text
    assert "Bias" in summary_text


@pytest.mark.requires_real_data
def test_forecast_benchmark_danra():
    """Verify ForecastBenchmark runs autoregressive evaluation on DANRA."""
    datastore = init_datastore_example("mdp")
    graph_name = "1level_danra_bench"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )

    predictor = GraphLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=2,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = ARForecaster(predictor, datastore)
    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
    )

    eval_steps = 3
    benchmark = ForecastBenchmark(
        datastore=datastore,
        split="test",
        eval_steps=eval_steps,
        batch_size=1,
        fss_kernel_sizes=[1, 3],
        device="cpu",
    )

    scorecard = benchmark.evaluate(model=model, max_batches=2)

    assert isinstance(scorecard, BenchmarkScorecard)
    assert scorecard.is_stable is True
    assert scorecard.num_rollout_steps == eval_steps
    assert scorecard.model_name == "GraphLAM"
    assert len(scorecard.variables) == len(datastore.get_vars_names("state"))
    assert scorecard.mean_step_latency_ms >= 0.0
    assert scorecard.vector_wind_rmse is not None
    assert len(scorecard.vector_wind_rmse) == eval_steps
    assert len(scorecard.wind_threshold_metrics) > 0

    # Research mode passes with advisory warnings
    passed_research, _ = scorecard.evaluate_verdict(mode="research")
    assert passed_research is True

    # Untrained candidate fails operational wind frequency bias gating
    passed_operational, msgs = scorecard.evaluate_verdict(mode="operational")
    assert passed_operational is False
    assert any("Wind Frequency Bias" in m for m in msgs)
