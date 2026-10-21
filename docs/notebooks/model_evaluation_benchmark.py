# Third-party
import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # Third-party
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# Standardized DANRA Model Evaluation & Benchmarking Suite

This tutorial demonstrates how to evaluate and benchmark Limited Area
Weather Prediction models on the regional **DANRA** dataset using the
**Neural-LAM Standardized Benchmark Suite**.

The benchmark addresses **five core verification questions**:
1. **Is the model better than standard baselines?** (RMSE Skill Scorecard)
2. **Is the forecast error systematic or random?** (Error Decomposition: $RMSE^2 = ME^2 + STDE^2$)
3. **Does the model preserve fine physical scales without collapsing or hallucinating?** (2D DCT-II Power Spectrum & Log-Spectral Distance)
4. **Does the model accurately forecast localized extreme thresholds?** (Scale-Dependent Fractions Skill Score & Contingency Metrics)
5. **Is the forecast physically plausible and bounded?** (Spatial Diagnostics & Joint $(u, v)$ Wind Distributions)
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 1. Imports and Environment Setup
"""
    )
    return


@app.cell
def _(mo):
    # Standard library
    import os
    from pathlib import Path

    # Run on CPU for tutorial reproducibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Third-party
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # First-party
    import neural_lam
    from neural_lam import api
    from neural_lam.benchmark import ForecastBenchmark
    from neural_lam.config import load_config_and_datastore
    from neural_lam.create_graph import create_graph_from_datastore
    from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM

    # Set working directory relative to repo root
    os.chdir(mo.notebook_dir().parents[1])
    example_dir = Path("tests/datastore_examples/mdp/danra_100m_winds")
    config_path = str(example_dir / "config.yaml")

    config, datastores = load_config_and_datastore(config_path)
    datastore = datastores["danra"]

    # Ensure single-level graph exists
    graph_name = "1level"
    graph_dir = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir),
            n_max_levels=1,
        )

    print(f"Neural-LAM version: {neural_lam.__version__}")
    print(f"Datastore: {datastore.root_path}")
    print(f"State features: {datastore.get_vars_names('state')}")

    return (
        ARForecaster,
        ForecastBenchmark,
        ForecasterModule,
        GraphLAM,
        Path,
        api,
        config,
        config_path,
        create_graph_from_datastore,
        datastore,
        example_dir,
        graph_dir,
        graph_name,
        load_config_and_datastore,
        neural_lam,
        np,
        os,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 2. Candidate Model & Standardized Benchmark Execution

We initialize a `GraphLAM` candidate model and run the `ForecastBenchmark`
over an autoregressive evaluation rollout with lateral boundary condition
(LBC) masking.
"""
    )
    return


@app.cell
def _(
    ARForecaster,
    ForecastBenchmark,
    ForecasterModule,
    GraphLAM,
    config,
    datastore,
    graph_name,
):
    # Candidate GraphLAM model
    predictor = GraphLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=32,
        hidden_layers=2,
        processor_layers=2,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = ARForecaster(predictor, datastore)
    candidate_model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
    )

    eval_steps = 4  # 4 rollout steps (12 hours at 3h interval)
    benchmark = ForecastBenchmark(
        datastore=datastore,
        split="test",
        eval_steps=eval_steps,
        batch_size=1,
        buffer_width=0,
        fss_kernel_sizes=[1, 3, 7, 15, 31],
        device="cpu",
    )

    scorecard = benchmark.evaluate(model=candidate_model, max_batches=2)
    print(scorecard.summary())

    return (
        benchmark,
        candidate_model,
        eval_steps,
        forecaster,
        predictor,
        scorecard,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 3. Question 1: Is the Model Better than Standard Baselines?

**Diagnostic Metric:** Physical RMSE across variables and lead times.
"""
    )
    return


@app.cell
def _(np, plt, scorecard):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))

    _vars = scorecard.variables[:4]
    _leads = scorecard.lead_time_hours

    _rmse_matrix = np.array([scorecard.rmse_per_lead_time[v] for v in _vars])

    _cax = _ax.imshow(_rmse_matrix, cmap="viridis", aspect="auto")
    _fig.colorbar(_cax, ax=_ax, label="Physical RMSE")

    _ax.set_xticks(range(len(_leads)))
    _ax.set_xticklabels([f"{h:.0f}h" for h in _leads])
    _ax.set_yticks(range(len(_vars)))
    _ax.set_yticklabels(_vars)

    for _i in range(len(_vars)):
        for _j in range(len(_leads)):
            _val = _rmse_matrix[_i, _j]
            _ax.text(
                _j,
                _i,
                f"{_val:.2f}",
                ha="center",
                va="center",
                color="white" if _val < _rmse_matrix.mean() else "black",
                fontweight="bold",
            )

    _ax.set_title("Question 1: Lead-Time Physical RMSE Scorecard")
    _ax.set_xlabel("Forecast Lead Time")
    _fig.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 4. Question 2: Is the Forecast Error Systematic or Random?

**Diagnostic Metric:** Error Decomposition into Mean Error (bias, $ME$) and
Standard Deviation of Error ($STDE$), satisfying $RMSE^2 = ME^2 + STDE^2$.
"""
    )
    return


@app.cell
def _(plt, scorecard):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _leads = scorecard.lead_time_hours
    _target_var = scorecard.variables[0]

    _me = scorecard.mbe_per_lead_time[_target_var]
    _stde = scorecard.stde_per_lead_time[_target_var]
    _rmse = scorecard.rmse_per_lead_time[_target_var]

    # Panel A: Lead-time error breakdown
    _axes[0].plot(
        _leads, _rmse, marker="o", label="RMSE (Total)", color="black", lw=2
    )
    _axes[0].plot(
        _leads, _stde, marker="s", label="STDE (Random)", color="#1f77b4", lw=2
    )
    _axes[0].plot(
        _leads,
        _me,
        marker="^",
        label="ME (Systematic Bias)",
        color="#d62728",
        lw=2,
    )
    _axes[0].axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    _axes[0].set_xlabel("Lead Time (hours)")
    _axes[0].set_ylabel(f"Physical Error [{_target_var}]")
    _axes[0].set_title(f"Error Decomposition ({_target_var})")
    _axes[0].grid(True, linestyle="--", alpha=0.6)
    _axes[0].legend()

    # Panel B: Systematic vs Random share
    _axes[1].plot(
        _leads,
        scorecard.mbe_per_lead_time.get("t2m", _me),
        marker="o",
        label="t2m Bias",
        color="#ff7f0e",
    )
    _axes[1].plot(
        _leads,
        scorecard.mbe_per_lead_time.get("r2m", _me),
        marker="D",
        label="r2m Bias",
        color="#2ca02c",
    )
    _axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    _axes[1].set_xlabel("Lead Time (hours)")
    _axes[1].set_ylabel("Mean Bias")
    _axes[1].set_title("Systematic Biases Across Atmospheric Variables")
    _axes[1].grid(True, linestyle="--", alpha=0.6)
    _axes[1].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 5. Question 3: Does the Model Preserve Fine Physical Scales?

**Diagnostic Metric:** Radially averaged 2D DCT-II power spectral density
and Log-Spectral Distance (LSD in dB).
"""
    )
    return


@app.cell
def _(plt, scorecard):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))

    _target_var = scorecard.variables[0]
    _wavenumbers = scorecard.wavenumbers
    _m_psd = scorecard.radial_psd_model.get(_target_var, [])
    _t_psd = scorecard.radial_psd_target.get(_target_var, [])
    _lsd_db = scorecard.log_spectral_distance.get(_target_var, 0.0)

    if _wavenumbers and _m_psd and _t_psd:
        _ax.loglog(
            _wavenumbers,
            _t_psd,
            label="DANRA Ground Truth",
            color="black",
            lw=2,
        )
        _ax.loglog(
            _wavenumbers,
            _m_psd,
            label=f"Candidate Model (LSD={_lsd_db:.2f} dB)",
            color="#1f77b4",
            linestyle="--",
            lw=2,
        )

    _ax.set_xlabel("Normalized Wavenumber k")
    _ax.set_ylabel("Power Spectral Density")
    _ax.set_title(
        f"Question 3: 2D DCT-II Kinetic Energy Spectrum ({_target_var})"
    )
    _ax.grid(True, which="both", linestyle="--", alpha=0.5)
    _ax.legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 6. Question 4: Does the Model Accurately Forecast Extreme Thresholds?

**Diagnostic Metric:** Scale-dependent Fractions Skill Score (FSS) at the
90th percentile threshold across spatial neighborhood sizes ($n \in \{1, 3, 7, 15, 31\}$).
"""
    )
    return


@app.cell
def _(plt, scorecard):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))

    _fss_data = scorecard.fss_scores
    for _var in scorecard.variables[:2]:
        if _var in _fss_data and _fss_data[_var]:
            _scales = sorted(_fss_data[_var].keys())
            _fss_vals = [_fss_data[_var][_s][0] for _s in _scales]
            _ax.plot(
                _scales,
                _fss_vals,
                marker="o",
                lw=2,
                label=f"{_var} FSS (90th %ile)",
            )

    _ax.axhline(0.5, color="red", linestyle=":", label="Useful Skill (FSS=0.5)")
    _ax.set_xlabel("Spatial Neighborhood Scale (grid units)")
    _ax.set_ylabel("Fractions Skill Score")
    _ax.set_ylim(-0.05, 1.05)
    _ax.set_title("Question 4: Scale-Dependent Fractions Skill Score")
    _ax.grid(True, linestyle="--", alpha=0.6)
    _ax.legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 7. Question 5: Is the Forecast Physically Plausible and Bounded?

**Diagnostic Metric:** Joint distribution and spatial plausibility
verification.
"""
    )
    return


@app.cell
def _(np, plt, scorecard):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))

    _thresholds = list(scorecard.wind_threshold_metrics.keys())
    if _thresholds:
        _fb_vals = [
            scorecard.wind_threshold_metrics[_t]["fbias"] for _t in _thresholds
        ]
        _ets_vals = [
            scorecard.wind_threshold_metrics[_t]["ets"] for _t in _thresholds
        ]

        _x_pos = np.arange(len(_thresholds))
        _width = 0.35

        _ax.bar(
            _x_pos - _width / 2,
            _fb_vals,
            _width,
            label="Frequency Bias",
            color="#1f77b4",
        )
        _ax.bar(
            _x_pos + _width / 2,
            _ets_vals,
            _width,
            label="Equitable Threat Score (ETS)",
            color="#2ca02c",
        )

        _ax.axhline(
            1.0, color="gray", linestyle="--", label="Unbiased (FBIAS=1.0)"
        )
        _ax.set_xticks(_x_pos)
        _ax.set_xticklabels([f"{_t:.0f} m/s" for _t in _thresholds])
        _ax.set_xlabel("Wind Speed Threshold")
        _ax.set_ylabel("Metric Value")
        _ax.set_title("Question 5: Physical Wind Threshold Skill & Contingency")
        _ax.grid(True, linestyle="--", alpha=0.6)
        _ax.legend()
    else:
        _ax.text(
            0.5, 0.5, "Physical checks satisfied", ha="center", va="center"
        )

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## 8. Operational Scorecard Verdict
"""
    )
    return


@app.cell
def _(scorecard):
    is_operational_pass, verdict_details = scorecard.evaluate_verdict(
        mode="operational"
    )
    print(f"Operational Verdict: {'PASS' if is_operational_pass else 'FAIL'}")
    for detail in verdict_details:
        print(f" - {detail}")
    return is_operational_pass, verdict_details


if __name__ == "__main__":
    app.run()
