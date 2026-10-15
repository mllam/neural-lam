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

    ### Prerequisites and Context

    > **Important:** This notebook is designed to be run from inside a **local clone** of [`mllam/neural-lam`](https://github.com/mllam/neural-lam). All paths are relative to the repository root.
    >
    > **Paper Reference:** For an in-depth context on the models used here, please refer to the paper \
        "Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings" \
        (Adamov et al., 2025) available at [arXiv:2504.09340](https://arxiv.org/abs/2504.09340).
    >
    > **Python Version:** Make sure you are using a Python version supported by the project (e.g. 3.10-3.14) with `ipykernel` installed.
    >
    > **Note on Future Graph Updates:** The graph generation step currently uses `create_graph.py`, but it will be migrated to use the upcoming `weather-model-graphs` package in the near future.
    """
    )
    return


@app.cell
def _():
    # Standard library
    import glob
    import os
    import subprocess
    import sys

    # Third-party
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from IPython.display import Image, display

    # First-party
    import neural_lam
    from neural_lam import utils
    from neural_lam.config import load_config_and_datastore
    from neural_lam.plot_graph import plot_graph

    return (
        Image,
        display,
        glob,
        load_config_and_datastore,
        neural_lam,
        np,
        os,
        plt,
        subprocess,
        sys,
        utils,
        xr,
    )


@app.cell
def _(os):
    # Verify we are at the repo root (should see neural_lam/, docs/, tests/, etc.)
    print("Current directory:", os.getcwd())
    print(
        "Repo contents:", [f for f in os.listdir(".") if not f.startswith(".")]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    """
    )
    return


@app.cell
def _(os):
    # The notebook is typically located in docs/notebooks/.
    # Change the working directory to the repository root so that paths like 'tests/...' resolve correctly.
    if os.getcwd().endswith("notebooks"):
        os.chdir("../../")
    print("Current working directory:", os.getcwd())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Environment Setup

    Set up the environment with [`uv`](https://docs.astral.sh/uv/) from the repository root (see the [README](https://github.com/mllam/neural-lam#installation)):

    ```bash
    uv sync --extra cpu --group dev
    ```

    This creates a `.venv` with the CPU build of PyTorch plus the development dependencies. For GPU runs swap `--extra cpu` for `--extra gpu` (CUDA 13.0) or `--extra gpu-cu128` (CUDA 12.8).

    Run the rest of this notebook on that environment's kernel (Python >=3.10 with `ipykernel` installed).
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

    > **Version requirement:** This notebook requires `mllam-data-prep >= 0.6.0`. Check with `python -m mllam_data_prep --version` or upgrade with `uv pip install --upgrade mllam-data-prep`.\n
    >\n
    > **Note:** This will download and process approximately 60–120 MB of data. It may take a few minutes on the first run.
    """
    )
    return


@app.cell
def _(subprocess, sys):
    # Run mllam-data-prep to create the processed DANRA .zarr dataset
    # Output will appear next to danra.datastore.yaml as danra.datastore.zarr
    #! {sys.executable} -m mllam_data_prep tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.yaml
    subprocess.call(
        [
            str(sys.executable),
            "-m",
            "mllam_data_prep",
            "tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.yaml",
        ]
    )
    return


@app.cell
def _(os):
    zarr_path = (
        "tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.zarr"
    )
    if os.path.exists(zarr_path):
        print("Zarr dataset at:", zarr_path)
    else:
        print("zarr dataset not found — check the output above for errors.")
    return


@app.cell
def _(np, plt, xr):
    ds = xr.open_zarr(
        "tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.zarr"
    )
    print(ds)
    if "state" in ds:
        da = ds["state"].isel(time=0, state_feature=0)
        values = da.values
        x = ds.coords["x"].values if "x" in ds.coords else None
        y = ds.coords["y"].values if "y" in ds.coords else None
        _fig, ax = plt.subplots(figsize=(8, 6))
        if x is not None and y is not None:
            x_unique = np.sort(np.unique(x))
            y_unique = np.sort(np.unique(y))
            xi = np.searchsorted(x_unique, x)
            yi = np.searchsorted(y_unique, y)
            grid = np.full((len(y_unique), len(x_unique)), np.nan)
            grid[yi, xi] = values
            pcm = ax.pcolormesh(
                x_unique, y_unique, grid, cmap="RdBu_r", shading="auto"
            )
            x_units = ds.coords["x"].attrs.get("units", "")
            y_units = ds.coords["y"].attrs.get("units", "")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        else:
            n = len(values)
            ny = int(np.sqrt(n))
            nx = n // ny
            grid = values[: ny * nx].reshape(ny, nx)
            pcm = ax.pcolormesh(grid, cmap="RdBu_r", shading="auto")
        feature_names = (
            list(ds["state_feature"].values)
            if "state_feature" in ds.coords
            else ["feature_0"]
        )
        ax.set_title(f"2D state field: {feature_names[0]} at t=0")
        plt.colorbar(pcm, ax=ax)
        plt.tight_layout()
        plt.show()
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
def _(subprocess, sys):
    # Generate the single-level graph for fast CPU execution
    # Graph files are stored in tests/datastore_examples/mdp/danra_100m_winds/graphs/1level/
    #! {sys.executable} -m neural_lam.create_graph --config_path tests/datastore_examples/mdp/danra_100m_winds/config.yaml --name 1level --levels 1
    subprocess.call(
        [
            str(sys.executable),
            "-m",
            "neural_lam.create_graph",
            "--config_path",
            "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
            "--name",
            "1level",
            "--levels",
            "1",
        ]
    )
    return


@app.cell
def _(glob):
    # Confirm the graph was created
    _graph_files = glob.glob(
        "tests/datastore_examples/mdp/danra_100m_winds/graph/1level/**",
        recursive=True,
    )
    print(f"Graph files created ({len(_graph_files)}):", _graph_files)
    return


@app.cell
def _(glob, os):
    _graph_dir = "tests/datastore_examples/mdp/danra_100m_winds/graph/1level"
    _graph_files = glob.glob(
        os.path.join(_graph_dir, "**", "*.pt"), recursive=True
    )
    if _graph_files:
        print(f"✅ Graph created with {len(_graph_files)} tensor file(s):")
        for f in sorted(_graph_files):
            size_kb = os.path.getsize(f) / 1024
            print(f"  {os.path.basename(f):40s} {size_kb:.1f} KB")
    else:
        print(
            "❌ No graph .pt files found — check the create_graph output above."
        )
    return


@app.cell
def _(load_config_and_datastore, np, os, plt, utils):
    # Third-party
    from IPython.display import HTML  # noqa: F401

    config_path = "tests/datastore_examples/mdp/danra_100m_winds/config.yaml"
    _, datastore = load_config_and_datastore(config_path=config_path)
    xy = datastore.get_xy("state", stacked=True)
    grid_xy_max_span = np.max(np.abs(xy))
    grid_pos = xy / grid_xy_max_span
    _graph_dir = os.path.join(datastore.root_path, "graph", "1level")
    hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=_graph_dir, mesh_node_features_scaling=grid_xy_max_span
    )
    _fig = plot_graph(
        grid_pos=grid_pos, hierarchical=hierarchical, graph_ldict=graph_ldict
    )
    _fig.write_html("graph_viz.html", include_plotlyjs="cdn")
    print(
        "Full interactive 3D graph saved to graph_viz.html - open it in a browser."
    )
    traces = {t.name: t for t in _fig.data}

    def edge_xyz(name, keep=None):
        """x, y, z of an edge trace (None-separated segments), optionally thinned."""
        t = traces[name]
        x = np.asarray(t.x, float)
        y = np.asarray(t.y, float)
        z = np.asarray(t.z, float)
        n_edges = x.size // 3
        if keep is not None and n_edges > keep:
            sel = np.zeros(x.size, dtype=bool)
            for i in range(0, n_edges, max(1, n_edges // keep)):
                sel[3 * i : 3 * i + 3] = True
            x, y, z = (x[sel], y[sel], z[sel])
        return (x, y, z)

    grid_x = np.asarray(traces["Grid nodes"].x, dtype=float)
    grid_y = np.asarray(traces["Grid nodes"].y, dtype=float)
    mesh_x = np.asarray(traces["Mesh nodes"].x, dtype=float)
    mesh_y = np.asarray(traces["Mesh nodes"].y, dtype=float)
    mesh_z = np.asarray(traces["Mesh nodes"].z, dtype=float)
    stride = max(1, grid_x.size // 20000)
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(*edge_xyz("G2M", keep=400), color="0.7", lw=0.2, alpha=0.4)
    ax3d.plot(
        *edge_xyz("M2M"), color="tab:blue", lw=0.4, alpha=0.6, label="M2M edges"
    )
    ax3d.scatter(
        grid_x[::stride],
        grid_y[::stride],
        np.zeros(grid_x[::stride].size),
        s=1,
        color="lightgray",
        label="grid nodes",
    )
    ax3d.scatter(
        mesh_x, mesh_y, mesh_z, s=10, color="tab:red", label="mesh nodes"
    )
    ax3d.set_title("Graph structure (3D: grid + mesh layers)")
    ax3d.set_zticks([])
    ax3d.legend(loc="upper right", markerscale=3)
    plt.show()
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

    **About logging:** By default, `neural-lam` logs via [Weights & Biases](https://wandb.ai). To suppress W&B upload during this demo, we set `WANDB_MODE=offline` so metrics are saved locally without requiring a W&B account or login.
    """
    )
    return


@app.cell
def _(os, subprocess, sys):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Force a non-interactive matplotlib backend in the subprocess:
    # the default TkAgg backend aborts (Tcl threading) when plotting
    # from Lightning callback threads.
    os.environ["MPLBACKEND"] = "Agg"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_lam.train_model",
            "--config_path",
            "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
            "--model",
            "graph_lam",
            "--graph",
            "1level",
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
        ],
        check=True,
    )
    return


@app.cell
def _(glob):
    # Find the checkpoint saved during training
    _ckpts = glob.glob("runs/**/*.ckpt", recursive=True)
    print("Checkpoints found:", _ckpts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Evaluation & Visualization

    Evaluation reuses the same `train_model` command with `--eval test` and the `--load` flag pointing to the checkpoint from training.

    Example predictions are plotted automatically and saved by the logger. Set `--n_example_pred` to control how many prediction plots to produce.
    """
    )
    return


@app.cell
def _(glob, os):
    _ckpts = glob.glob("runs/**/*.ckpt", recursive=True)
    if not _ckpts:
        raise FileNotFoundError(
            "No checkpoint found under runs/ - the training cell above did not complete."
        )
    else:
        ckpt_path = max(_ckpts, key=os.path.getmtime)
        print(f"✅ Using checkpoint: {ckpt_path}")
    return (ckpt_path,)


@app.cell
def _():
    # Standard library
    import argparse

    # Third-party
    import torch

    # First-party
    from neural_lam.config import (
        DatastoreSelection,
        ManualStateFeatureWeighting,
        NeuralLAMConfig,
        OutputClamping,
        TrainingConfig,
        UniformFeatureWeighting,
    )

    torch.serialization.add_safe_globals(
        [
            argparse.Namespace,
            DatastoreSelection,
            ManualStateFeatureWeighting,
            NeuralLAMConfig,
            OutputClamping,
            TrainingConfig,
            UniformFeatureWeighting,
        ]
    )
    return


@app.cell
def _(ckpt_path, os, subprocess, sys):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_lam.train_model",
            "--config_path",
            "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
            "--model",
            "graph_lam",
            "--graph",
            "1level",
            "--eval",
            "test",
            "--load",
            ckpt_path,
            "--processor_layers",
            "2",
            "--n_example_pred",
            "2",
            "--ar_steps_eval",
            "4",
            "--num_workers",
            "0",
            "--val_steps_to_log",
            "1",
        ],
        check=True,
    )
    return


@app.cell
def _(Image, display, glob, os):
    rmse_plots = sorted(
        glob.glob("runs/**/test_rmse_*.png", recursive=True),
        key=os.path.getmtime,
    )
    if rmse_plots:
        print("RMSE scorecard:", rmse_plots[-1])
        display(Image(filename=rmse_plots[-1]))
    else:
        print("test_rmse plot not found - check eval output above.")

    example_plots = sorted(
        glob.glob("runs/**/*_example_*.png", recursive=True),
        key=os.path.getmtime,
    )
    if example_plots:
        n_show = min(2, len(example_plots))
        print(f"Showing {n_show} of {len(example_plots)} prediction plot(s):")
        for path in example_plots[-n_show:]:
            print(" ", path)
            display(Image(filename=path))
    else:
        print("No prediction plots found - check eval output above.")
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

    Re-create the environment with a CUDA build of PyTorch and `--devices` will auto-detect your GPU:
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

    Remove the `WANDB_MODE=offline` line (or set it to `online`) and optionally set `--logger wandb --logger-project <your-project-name>`.

    ### 🔗 Further reading

    - [neural-lam README](https://github.com/mllam/neural-lam)
    - [mllam-data-prep](https://github.com/mllam/mllam-data-prep)
    - [Graph-based Neural Weather Prediction (NeurIPS 2023)](https://arxiv.org/abs/2309.17370)
    - [Probabilistic Weather Forecasting with Hierarchical GNNs (NeurIPS 2024)](https://arxiv.org/abs/2406.04759)
    """
    )
    return


if __name__ == "__main__":
    app.run()
