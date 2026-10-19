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
    # Hello World: Training `neural-lam` on DANRA

    This notebook provides a **beginner-friendly, end-to-end walkthrough** for running a minimal model training pipeline in `neural-lam` using a small, public DANRA dataset. Every step uses the Python API in `neural_lam.api`.

    It covers:
    1. Environment setup (CPU-safe)
    2. Data preprocessing with `mllam-data-prep`
    3. Graph generation (single-level, for speed)
    4. Training for 1 epoch on CPU
    5. Evaluation and example predictions
    6. Scaling tips for bigger runs

    ---

    ## Prerequisites and Context

    > **Important:** This notebook is designed to be run from inside a **local clone** of [`mllam/neural-lam`](https://github.com/mllam/neural-lam). All paths are relative to the repository root.
    >
    > **Paper Reference:** For an in-depth context on the models used here, please refer to the paper \
        "Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings" \
        (Adamov et al., 2025) available at [arXiv:2504.09340](https://arxiv.org/abs/2504.09340).
    >
    > **Python Version:** Make sure you are using a Python version supported by the project (3.10-3.14).
    """
    )
    return


@app.cell
def _(mo):
    # Standard library
    import os
    from pathlib import Path

    # Run on CPU and keep W&B metrics local, before torch and wandb start
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["WANDB_MODE"] = "offline"

    # Third-party
    import fsspec
    import matplotlib
    import mllam_data_prep
    import numpy as np
    import xarray as xr

    # Lightning callbacks plot from worker threads, which GUI backends
    # such as TkAgg do not support
    matplotlib.use("Agg")
    # Third-party
    import matplotlib.pyplot as plt

    # First-party
    import neural_lam
    from neural_lam import api, utils
    from neural_lam.config import load_config_and_datastore
    from neural_lam.plot_graph import plot_graph

    # All paths below are relative to the repository root
    os.chdir(mo.notebook_dir().parents[1])
    example_dir = Path("tests/datastore_examples/mdp/danra_100m_winds")
    config_path = str(example_dir / "config.yaml")
    print("Working directory:", Path.cwd())
    return (
        api,
        config_path,
        example_dir,
        fsspec,
        load_config_and_datastore,
        mllam_data_prep,
        neural_lam,
        np,
        plot_graph,
        plt,
        utils,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Environment Setup

    Set up the environment with [`uv`](https://docs.astral.sh/uv/) from the repository root (see "Installing Neural-LAM" in the [README](https://github.com/mllam/neural-lam/blob/main/README.md)) and open this notebook with marimo:

    ```bash
    uv sync --extra cpu --extra docs --group dev
    source .venv/bin/activate
    marimo edit docs/notebooks/hello_world_danra.py
    ```

    This creates a `.venv` with the CPU build of PyTorch, marimo (from the `docs` extra) and the development dependencies. For GPU runs swap `--extra cpu` for `--extra gpu` (CUDA 13.0) or `--extra gpu-cu128` (CUDA 12.8).

    To run the whole tutorial non-interactively use `python docs/notebooks/hello_world_danra.py`.
    """
    )
    return


@app.cell
def _(neural_lam):
    # Verify the install
    print("neural-lam version:", neural_lam.__version__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Data Configuration & Preprocessing

    We use the **DANRA 100m winds example** already in this repo at:
    ```
    tests/datastore_examples/mdp/danra_100m_winds/
      ├── config.yaml          ← neural-lam configuration
      └── danra.datastore.yaml ← mllam-data-prep datastore config
    ```

    This example uses a **cropped DANRA dataset** (~100×100 grid points, ~10 days in April 2022) served from a public ECMWF object store — no local data download is required.

    `mllam_data_prep.create_dataset_zarr` reads the datastore config, fetches the data, and writes a processed `.zarr` archive next to it.

    > **Note:** This will download and process approximately 60–120 MB of data. It may take a few minutes on the first run.
    """
    )
    return


@app.cell
def _(example_dir, fsspec, mllam_data_prep, xr):
    _zarr_path = example_dir / "danra.datastore.zarr"
    if not _zarr_path.exists():
        # zarr 3 reads the remote inputs through an async https filesystem
        fsspec.config.conf["https"] = {"asynchronous": True}
        mllam_data_prep.create_dataset_zarr(
            fp_config=example_dir / "danra.datastore.yaml"
        )
    else:
        print(f"Dataset already exists at {_zarr_path}, skipping preparation.")
    ds = xr.open_zarr(_zarr_path)
    ds
    return (ds,)


@app.cell
def _(ds, np, plt):
    _da = ds["state"].isel(time=0, state_feature=0)
    _x_unique = np.unique(ds["x"].values)
    _y_unique = np.unique(ds["y"].values)
    _grid = np.full((len(_y_unique), len(_x_unique)), np.nan)
    _grid[
        np.searchsorted(_y_unique, ds["y"].values),
        np.searchsorted(_x_unique, ds["x"].values),
    ] = _da.values
    fig, _ax = plt.subplots(figsize=(8, 6))
    _pcm = _ax.pcolormesh(
        _x_unique, _y_unique, _grid, cmap="RdBu_r", shading="auto"
    )
    _ax.set_xlabel("x (m)")
    _ax.set_ylabel("y (m)")
    _ax.set_title(f"{_da['state_feature'].item()} at t=0")
    fig.colorbar(_pcm, ax=_ax)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Graph Generation

    `neural-lam` uses a **graph** to define the message-passing structure of the GNN. `api.create_graph` builds it with [`weather-model-graphs`](https://github.com/mllam/weather-model-graphs) from one of these archetypes:

    | Archetype      | Mesh                       | Model                  |
    |----------------|----------------------------|------------------------|
    | `keisler`      | Single level               | `graph_lam` (this notebook) |
    | `graphcast`    | Multi-scale, flattened     | `graph_lam`            |
    | `hierarchical` | Hierarchy of mesh levels   | `hi_lam`, `hi_lam_parallel` |

    For this Hello World example we use the **single-level `keisler` graph** — the lightest option, ideal for CPU.
    """
    )
    return


@app.cell
def _(api, config_path, example_dir):
    graph_name = "1level"
    api.create_graph(
        config_path=config_path, name=graph_name, archetype="keisler"
    )
    graph_dir = example_dir / "graph" / graph_name
    for _f in sorted(graph_dir.glob("*.pt")):
        print(f"  {_f.name:40s} {_f.stat().st_size / 1024:.1f} KB")
    return graph_dir, graph_name


@app.cell
def _(config_path, graph_dir, load_config_and_datastore, np, plot_graph, utils):
    # Interactive 3D view of the grid and mesh nodes and their connections
    _, _datastore, _ = load_config_and_datastore(config_path=config_path)
    _xy = _datastore.get_xy("state", stacked=True)
    _grid_xy_max_span = np.max(np.abs(_xy))
    _hierarchical, _graph_ldict = utils.load_graph(
        graph_dir_path=str(graph_dir),
        mesh_node_features_scaling=_grid_xy_max_span,
    )
    plot_graph(
        grid_pos=_xy / _grid_xy_max_span,
        hierarchical=_hierarchical,
        graph_ldict=_graph_ldict,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Training on CPU

    `api.train` takes the same options as `python -m neural_lam.train_model`, as keyword arguments. We train the `graph_lam` model for **1 epoch** on the single-level graph:

    | Argument | Value | Why |
    |------|-------|-----|
    | `model` | `"graph_lam"` | Simplest model — compatible with non-hierarchical graphs |
    | `graph` | `"1level"` | Must match the graph name used in the previous step |
    | `epochs` | `1` | Minimal run — just to verify the pipeline works end-to-end |
    | `processor_layers` | `2` | Reduced from default (4) for CPU |
    | `ar_steps_train` | `1` | Unroll 1 time-step during training — reduces memory and compute |
    | `ar_steps_eval` | `1` | Also 1 step for the val pass within this demo run |

    It returns a `Run` handle with the run directory and the checkpoint path.

    **About logging:** By default, `neural-lam` logs via [Weights & Biases](https://wandb.ai). To suppress W&B upload during this demo, the import cell sets `WANDB_MODE=offline` so metrics are saved locally without requiring a W&B account or login.
    """
    )
    return


@app.cell
def _(api, config_path, graph_name):
    train_run = api.train(
        config_path=config_path,
        model="graph_lam",
        graph=graph_name,
        epochs=1,
        processor_layers=2,
        ar_steps_train=1,
        ar_steps_eval=1,
        num_workers=0,
        val_steps_to_log=[1],
    )
    print("Checkpoint:", train_run.checkpoint_path)
    return (train_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Evaluation & Visualization

    `api.evaluate` loads the checkpoint from training and runs the test set. The returned `Run` points to the plots it writes: the test RMSE per lead time (`rmse_plot`) and, with `create_gif=True`, a GIF per variable for each of the `n_example_pred` example forecasts (`example_plots(i)`).
    """
    )
    return


@app.cell
def _(api, config_path, graph_name, train_run):
    eval_run = api.evaluate(
        config_path=config_path,
        load=str(train_run.checkpoint_path),
        model="graph_lam",
        graph=graph_name,
        processor_layers=2,
        ar_steps_eval=4,
        n_example_pred=1,
        create_gif=True,
        num_workers=0,
        val_steps_to_log=[1],
    )
    return (eval_run,)


@app.cell
def _(eval_run, mo):
    mo.vstack(
        [
            mo.pdf(src=eval_run.rmse_plot, height="50vh"),
            *[
                mo.image(src=_gif)
                for _gif in sorted(eval_run.example_plots(1).glob("*.gif"))
            ],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Scaling Tips & Next Steps

    ### 🔁 Use a larger / hierarchical graph

    For a production-quality run, switch to the hierarchical graph and the Hi-LAM model:

    ```python
    api.create_graph(
        config_path="<your_config.yaml>",
        name="hierarchical",
        archetype="hierarchical",
    )
    api.train(
        config_path="<your_config.yaml>",
        model="hi_lam",
        graph="hierarchical",
        epochs=200,
    )
    ```

    The same steps are available on the command line as `python -m neural_lam.create_graph_with_wmg` and `python -m neural_lam.train_model`.

    ### ⚡ Enable GPU training

    Re-create the environment with a CUDA build of PyTorch and remove the `CUDA_VISIBLE_DEVICES` line from the import cell, so that `devices` (default `auto`) picks up your GPU:
    ```bash
    uv sync --extra gpu        # CUDA 13.0
    # or: uv sync --extra gpu-cu128   # CUDA 12.8
    ```

    ### 📦 Use larger / full DANRA data

    Modify `danra.datastore.yaml` to extend the `coord_ranges.time` window, or point `inputs[*].path` at a larger dataset. See the [mllam-data-prep README](https://github.com/mllam/mllam-data-prep) for full configuration options, including parallel processing of large datasets with dask.

    ### 📊 Enable cloud logging with W&B

    Remove the `WANDB_MODE` line in the import cell (or set it to `online`) and optionally pass `logger="wandb", logger_project="<your-project-name>"`.

    ### 🔗 Further reading

    - [neural-lam README](https://github.com/mllam/neural-lam)
    - [mllam-data-prep](https://github.com/mllam/mllam-data-prep)
    - [Graph-based Neural Weather Prediction for Limited Area Modeling (NeurIPS 2023 CCAI workshop)](https://arxiv.org/abs/2309.17370)
    - [Probabilistic Weather Forecasting with Hierarchical GNNs (NeurIPS 2024)](https://arxiv.org/abs/2406.04759)
    """
    )
    return


if __name__ == "__main__":
    app.run()
