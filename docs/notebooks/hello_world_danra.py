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

    This notebook provides a **beginner-friendly, end-to-end walkthrough** for running a minimal model training pipeline in `neural-lam` using a small, public DANRA dataset.

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
    >
    > **Note on Future Graph Updates:** The graph generation step currently uses `create_graph.py`, but it will be migrated to use the upcoming `weather-model-graphs` package in the near future.
    """
    )
    return


@app.cell
def _(mo):
    # Standard library
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Third-party
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    # First-party
    import neural_lam
    from neural_lam import utils
    from neural_lam.config import load_config_and_datastore
    from neural_lam.plot_graph import plot_graph

    # All paths below are relative to the repository root
    os.chdir(mo.notebook_dir().parents[1])
    example_dir = Path("tests/datastore_examples/mdp/danra_100m_winds")
    config_path = str(example_dir / "config.yaml")
    print("Working directory:", Path.cwd())
    return (
        Path,
        config_path,
        example_dir,
        load_config_and_datastore,
        neural_lam,
        np,
        os,
        plot_graph,
        plt,
        subprocess,
        sys,
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

    The `mllam-data-prep` command reads the datastore config, fetches the data, and writes a processed `.zarr` archive to disk.

    > **Note:** This will download and process approximately 60–120 MB of data. It may take a few minutes on the first run.
    """
    )
    return


@app.cell
def _(example_dir, os, subprocess, sys, xr):
    # Run mllam-data-prep to create the processed DANRA .zarr dataset
    # next to danra.datastore.yaml, unless it already exists
    _zarr_path = example_dir / "danra.datastore.zarr"
    if not _zarr_path.exists():
        _env = os.environ.copy()
        _env["FSSPEC_HTTPS_ASYNCHRONOUS"] = "true"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mllam_data_prep",
                str(example_dir / "danra.datastore.yaml"),
            ],
            check=True,
            env=_env,
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

    `neural-lam` uses a **graph** to define the message-passing structure of the GNN. Different graph types exist:

    | Graph type  | Command flag                    | Use case               |
    |-------------|----------------------------------|------------------------|
    | L1-LAM      | `--name 1level --levels 1`       | Quick demo (this notebook) |
    | GC-LAM      | `--name multiscale`              | Standard multi-scale model |
    | Hi-LAM      | `--name hierarchical --hierarchical` | Hierarchical model (production) |

    For this Hello World example we use the **L1 (single-level) graph** — the lightest option, ideal for CPU.
    """
    )
    return


@app.cell
def _(config_path, example_dir, os, subprocess, sys):
    # Generate the single-level graph for fast CPU execution
    graph_name = "1level"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_lam.create_graph",
            "--config_path",
            config_path,
            "--name",
            graph_name,
            "--levels",
            "1",
        ],
        check=True,
    )
    graph_dir = example_dir / "graph" / graph_name
    for _f in sorted(graph_dir.glob("*.pt")):
        print(f"  {_f.name:40s} {os.path.getsize(_f) / 1024:.1f} KB")
    return graph_dir, graph_name


@app.cell
def _(config_path, graph_dir, load_config_and_datastore, np, plot_graph, utils):
    # Interactive 3D view of the grid and mesh nodes and their connections
    _, _datastore = load_config_and_datastore(config_path=config_path)
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

    We train the `graph_lam` model for **1 epoch** using the L1 graph. Key flags used here:

    | Flag | Value | Why |
    |------|-------|-----|
    | `--model` | `graph_lam` | Simplest model — compatible with non-hierarchical graphs |
    | `--graph` | `1level` | Must match the graph name used in the previous step |
    | `--epochs` | `1` | Minimal run — just to verify the pipeline works end-to-end |
    | `--processor_layers` | `2` | Reduced from default (4) for CPU |
    | `--ar_steps_train` | `1` | Unroll 1 time-step during training — reduces memory and compute |
    | `--ar_steps_eval` | `1` | Also 1 step for the val pass within this demo run |
    | `--logger_run_name` | `hello_world_danra_train` | Fixed run directory under `runs/`, so the checkpoint path is known |

    **About logging:** By default, `neural-lam` logs via [Weights & Biases](https://wandb.ai). To suppress W&B upload during this demo, we set `WANDB_MODE=offline` so metrics are saved locally without requiring a W&B account or login.
    """
    )
    return


@app.cell
def _(Path, config_path, graph_name, os, subprocess, sys):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Force a non-interactive matplotlib backend in the subprocess:
    # the default TkAgg backend aborts (Tcl threading) when plotting
    # from Lightning callback threads.
    os.environ["MPLBACKEND"] = "Agg"

    _run_name = "hello_world_danra_train"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_lam.train_model",
            "--config_path",
            config_path,
            "--model",
            "graph_lam",
            "--graph",
            graph_name,
            "--epochs",
            "1",
            "--processor_layers",
            "2",
            "--ar_steps_train",
            "1",
            "--ar_steps_eval",
            "1",
            "--num_workers",
            "0",
            "--val_steps_to_log",
            "1",
            "--logger_run_name",
            _run_name,
        ],
        check=True,
    )
    ckpt_path = Path("runs") / _run_name / "checkpoints" / "last.ckpt"
    print("Checkpoint:", ckpt_path)
    return (ckpt_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Evaluation & Visualization

    Evaluation reuses the same `train_model` command with `--eval test` and the `--load` flag pointing to the checkpoint from training.

    The test RMSE per lead time is saved as `test_rmse.pdf` in the run directory. With `--create_gif`, `--n_example_pred` example forecasts are also written there as per-variable GIFs.
    """
    )
    return


@app.cell
def _(Path, ckpt_path, config_path, graph_name, subprocess, sys):
    _run_name = "hello_world_danra_eval"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_lam.train_model",
            "--config_path",
            config_path,
            "--model",
            "graph_lam",
            "--graph",
            graph_name,
            "--eval",
            "test",
            "--load",
            str(ckpt_path),
            "--processor_layers",
            "2",
            "--n_example_pred",
            "1",
            "--create_gif",
            "--ar_steps_eval",
            "4",
            "--num_workers",
            "0",
            "--val_steps_to_log",
            "1",
            "--logger_run_name",
            _run_name,
        ],
        check=True,
    )
    eval_dir = Path("runs") / _run_name
    return (eval_dir,)


@app.cell
def _(eval_dir, mo):
    mo.vstack(
        [
            mo.pdf(src=eval_dir / "test_rmse.pdf", height="50vh"),
            *[mo.image(src=_gif) for _gif in sorted(eval_dir.glob("*/*.gif"))],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Scaling Tips & Next Steps

    ### 🔁 Use a larger / hierarchical graph

    For a production-quality run, switch to the hierarchical Hi-LAM graph:

    ```bash
    python -m neural_lam.create_graph \
        --config_path <your_config.yaml> \
        --name hierarchical \
        --hierarchical

    python -m neural_lam.train_model \
        --config_path <your_config.yaml> \
        --model hi_lam \
        --graph hierarchical \
        --epochs 200
    ```

    ### ⚡ Enable GPU training

    Re-create the environment with a CUDA build of PyTorch and remove the `CUDA_VISIBLE_DEVICES=""` line from the training cell, so that `--devices` (default `auto`) picks up your GPU:
    ```bash
    uv sync --extra gpu        # CUDA 13.0
    # or: uv sync --extra gpu-cu128   # CUDA 12.8
    ```

    ### 📦 Use larger / full DANRA data

    Modify `danra.datastore.yaml` to extend the `coord_ranges.time` window, or point `inputs[*].path` at a larger dataset. See the [mllam-data-prep README](https://github.com/mllam/mllam-data-prep) for full configuration options.

    For large datasets (≥10 GB), use parallel preprocessing:
    ```bash
    python -m mllam_data_prep \
        <your_datastore.yaml> \
        --dask-distributed-local-core-fraction 0.5
    ```

    ### 📊 Enable cloud logging with W&B

    Remove the `WANDB_MODE=offline` line in the training cell (or set it to `online`) and optionally set `--logger wandb --logger-project <your-project-name>`.

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
