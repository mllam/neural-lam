"""Tests for the forecast benchmark evaluation suite."""

# Standard library
from pathlib import Path

# First-party
from neural_lam import config as nlconfig
from neural_lam.benchmark import BenchmarkScorecard, ForecastBenchmark
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from tests.conftest import init_datastore_example


def test_forecast_benchmark_scorecard_and_evaluation():
    """Verify ForecastBenchmark runs autoregressive evaluation on DANRA."""
    datastore = init_datastore_example("mdp")
    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
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
        device="cpu",
    )

    scorecard = benchmark.evaluate(model=model, max_batches=2)

    assert isinstance(scorecard, BenchmarkScorecard)
    assert scorecard.is_stable is True
    assert scorecard.num_rollout_steps == eval_steps
    assert scorecard.model_name == "ARForecaster"
    assert len(scorecard.variables) == len(datastore.get_vars_names("state"))
    assert scorecard.mean_step_latency_ms >= 0.0

    for var in scorecard.variables:
        mse_lead = scorecard.mse_per_lead_time[var]
        rmse_lead = scorecard.rmse_per_lead_time[var]
        bias_lead = scorecard.mbe_per_lead_time[var]
        assert len(mse_lead) == eval_steps
        assert len(rmse_lead) == eval_steps
        assert len(bias_lead) == eval_steps
        assert all(isinstance(val, float) and val >= 0.0 for val in mse_lead)
        assert all(isinstance(val, float) and val >= 0.0 for val in rmse_lead)
        assert len(scorecard.radial_psd[var]) == len(scorecard.wavenumbers)

    summary_text = scorecard.summary()
    assert "Forecast Benchmark Summary" in summary_text
    assert "RMSE" in summary_text
    assert "Bias" in summary_text
