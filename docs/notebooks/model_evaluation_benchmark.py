# Third-party
import marimo

__generated_with = "0.24.2"
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
    # Model Forecast Evaluation & Benchmarking on DANRA

    This tutorial demonstrates how to evaluate, compare, and benchmark Limited
    Area Weather Prediction models using the **Neural-LAM Standardized
    Benchmark Suite** on the **DANRA** regional dataset.

    ### State-of-the-Art Limited Area Verification
    When evaluating candidate architectures (Deterministic GraphLAM/HiLAM,
    Conditional Flow Matching (CFM), CNNs, or Transformers), regional domain
    verification requires:
    1. **Lateral Boundary Condition (LBC) Buffer Masking**: Isolating interior
       regional dynamics from external driving boundary relaxation.
    2. **2D Discrete Cosine Transform (DCT-II) Spectral Decomposition**:
       Avoiding periodic FFT edge-step discontinuities to evaluate true
       atmospheric turbulence energy decay ($k^{-3}$ synoptic, $k^{-5/3}$
       mesoscale).
    3. **Scale-Dependent Fractions Skill Score (FSS)**: Eliminating the
       double-penalty displacement problem for localized high-impact weather
       features (e.g. wind gusts, precipitation).
    4. **Spectral Collapse Ratio (SCR) & Hallucination Index (HI)**:
       Diagnosing spatial blurring in deep autoregressive MSE models vs
       realistic small-scale variance generation in CFM models.
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
def _():
    # Standard library
    from pathlib import Path

    # Third-party
    import matplotlib.pyplot as plt

    # First-party
    import neural_lam
    from neural_lam import config as nlconfig
    from neural_lam.benchmark import ForecastBenchmark
    from neural_lam.create_graph import create_graph_from_datastore
    from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
    from tests.conftest import init_datastore_example

    print(f"Neural-LAM version: {neural_lam.__version__}")
    return (
        ARForecaster,
        ForecastBenchmark,
        ForecasterModule,
        GraphLAM,
        Path,
        create_graph_from_datastore,
        init_datastore_example,
        nlconfig,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Load the Standard DANRA Regional Datastore & Build Graph
    """
    )
    return


@app.cell
def _(Path, create_graph_from_datastore, init_datastore_example):
    # Load standard DANRA cropped datastore (100m winds, 2m temp & humidity)
    datastore = init_datastore_example("mdp")
    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    # Ensure graph exists
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    print(f"Datastore root: {datastore.root_path}")
    grid_y = datastore.grid_shape_state.y
    grid_x = datastore.grid_shape_state.x
    print(f"Grid shape: {grid_y}x{grid_x}")
    print(f"State features: {datastore.get_vars_names('state')}")
    return datastore, graph_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Instantiate Model Architecture

    We instantiate a `GraphLAM` model wrapped in `ARForecaster` and
    `ForecasterModule`.
    """
    )
    return


@app.cell
def _(
    ARForecaster,
    ForecasterModule,
    GraphLAM,
    datastore,
    graph_name,
    nlconfig,
):
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    predictor = GraphLAM(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=32,
        hidden_layers=2,
        processor_layers=3,
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
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Run the Standardized Benchmark Suite

    The `ForecastBenchmark` runs multi-step autoregressive rollouts on the test
    set, computing physical errors, biases, step latencies, 2D DCT-II power
    spectra, and scale-dependent Fractions Skill Scores.
    """
    )
    return


@app.cell
def _(ForecastBenchmark, datastore, model):
    eval_steps = 4  # Evaluate 4 rollout steps (12 hours at 3-hour step length)
    benchmark = ForecastBenchmark(
        datastore=datastore,
        split="test",
        eval_steps=eval_steps,
        batch_size=1,
        buffer_width=10,
        fss_kernel_sizes=[1, 3, 7],
        device="cpu",
    )

    scorecard = benchmark.evaluate(model=model, max_batches=2)
    print(scorecard.summary())
    return (scorecard,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Diagnostic Visualizations: Multi-Metric Verification Suite

    We plot the three core atmospheric verification dimensions:
    1. **Physical Lead-Time Errors**: Interior RMSE and Bias growth over rollout
       steps.
    2. **Scale-Dependent Fractions Skill Score (FSS)**: Spatial tolerance skill
       curve (Roberts & Lean 2008) resolving the double-penalty problem.
    3. **2D DCT-II Power Spectral Density**: Energy spectrum (Denis et al. 2002)
       diagnosing fine-scale turbulence preservation vs spatial collapse.
    """
    )
    return


@app.cell
def _(plt, scorecard):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Lead-Time RMSE & Bias Plot
    for var in scorecard.variables[:2]:
        axes[0].plot(
            scorecard.lead_time_hours,
            scorecard.rmse_per_lead_time[var],
            marker="o",
            label=f"{var} RMSE",
        )
        axes[0].plot(
            scorecard.lead_time_hours,
            scorecard.mbe_per_lead_time[var],
            marker="^",
            linestyle="--",
            alpha=0.7,
            label=f"{var} Bias",
        )
    axes[0].axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Forecast Lead Time (hours)")
    axes[0].set_ylabel("Physical Units")
    axes[0].set_title("Lead-Time RMSE & Bias (Interior Domain)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    # 2. Scale-Dependent Fractions Skill Score (FSS)
    fss_plotted = False
    for var in scorecard.variables[:2]:
        if var in scorecard.fss_scores and scorecard.fss_scores[var]:
            scales = sorted(scorecard.fss_scores[var].keys())
            fss_vals = [scorecard.fss_scores[var][s][0] for s in scales]
            axes[1].plot(
                scales,
                fss_vals,
                marker="D",
                label=f"{var} FSS",
            )
            fss_plotted = True
    if fss_plotted:
        axes[1].axhline(
            0.5,
            color="red",
            linestyle=":",
            alpha=0.8,
            label="Useful Skill (FSS=0.5)",
        )
        axes[1].set_xlabel("Spatial Neighborhood Scale (grid units)")
        axes[1].set_ylabel("FSS (75th Percentile)")
        axes[1].set_title("Scale-Dependent Fractions Skill Score")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(fontsize=9)

    # 3. 2D DCT-II Radial Power Spectral Density
    if scorecard.wavenumbers:
        for var in scorecard.variables[:2]:
            axes[2].loglog(
                scorecard.wavenumbers,
                scorecard.radial_psd[var],
                marker="s",
                label=f"{var} DCT-II PSD",
            )
        axes[2].set_xlabel("Normalized Wavenumber k")
        axes[2].set_ylabel("Power Spectral Density")
        axes[2].set_title("2D DCT-II Kinetic Energy Spectrum")
        axes[2].grid(True, which="both", linestyle="--", alpha=0.6)
        axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
