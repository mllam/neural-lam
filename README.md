[![slack](https://img.shields.io/badge/slack-join-brightgreen.svg?logo=slack)](https://kutt.to/mllam)
[![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml)
[![CPU+GPU testing](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml)

<p align="middle">
    <img src="https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
Built on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), and [PyG](https://pyg.org/), it provides modular, swappable components for data, graphs, and models.

It is aimed at researchers and engineers working on ML-based numerical weather prediction who want to train, evaluate, or extend GNN-based LAM models on their own data.

## Documentation

Full guides, tutorials, and API reference live at **[mllam.github.io/neural-lam](https://mllam.github.io/neural-lam/)**:

- 📦 [Installation](https://mllam.github.io/neural-lam/getting-started/installation.html)
- 🚀 [Quickstart](https://mllam.github.io/neural-lam/getting-started/quickstart.html) and the [DANRA tutorial](https://mllam.github.io/neural-lam/notebooks/hello_world_danra.html)
- 📖 User guide: [configuration and datastores](https://mllam.github.io/neural-lam/user-guide/datastores.html), [graph creation](https://mllam.github.io/neural-lam/user-guide/graphs.html), [training and evaluation](https://mllam.github.io/neural-lam/user-guide/training.html), [experiment logging](https://mllam.github.io/neural-lam/user-guide/logging.html)
- 📚 [API Reference](https://mllam.github.io/neural-lam/api/index.html)
- 🗺️ [Graph Storage Specification](https://mllam.github.io/neural-lam/graph_storage_spec.html)

# Modularity

Models, graphs, and data are kept independent so each component can be swapped without changing the others.

<p align="middle">
  <img src="https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_setup.png" width="600"/>
</p>

The three implemented architectures are Graph-LAM (flat), Hi-LAM (hierarchical), and Hi-LAM-Parallel.
Global forecasting is available on a [separate branch](https://github.com/mllam/neural-lam/tree/prob_model_global) ([planned to be merged](https://github.com/mllam/neural-lam/issues/63)).

# Installing Neural-LAM

```bash
python -m neural_lam.create_graph_with_wmg --config_path <neural-lam-config-path> --archetype <archetype>
```

Available archetypes:

* **keisler** (default): `python -m neural_lam.create_graph_with_wmg --config_path <neural-lam-config-path> --archetype keisler`
* **graphcast**: `python -m neural_lam.create_graph_with_wmg --config_path <neural-lam-config-path> --archetype graphcast`
* **hierarchical**: `python -m neural_lam.create_graph_with_wmg --config_path <neural-lam-config-path> --archetype hierarchical`

Run `python -m neural_lam.create_graph_with_wmg --help` for the full list of
options (e.g. `--mesh_node_distance`, `--mesh_grid_distance_ratio`,
`--level_refinement_factor`, `--max_num_levels`).

## Logging your experiments

### Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally under the run directory (e.g. `runs/<run-name>/wandb/`).
The W&B project name is set to `neural-lam`, but this can be changed in the flags of `python -m neural_lam.train_model` (using argsparse).
See the [W&B documentation](https://docs.wandb.ai/) for details.

If you would like to login and use W&B, run:
```
wandb login
```
If you would like to turn off W&B and just log things locally, run:
```
wandb off
```

### MLFlow Integration
The project is also integrated with [MLFlow](https://mlflow.org/) for logging and storing artefacts.

MLFlow is not used by default, but can be switched to by setting `--logger mlflow` in the training command. With MLFlow enabled, training configuration, training/test statistics and plots are logged to the MLFlow server. MLFlow is self-hosted and can be run locally or on a server. See the [MLFlow documentation](https://mlflow.org/docs/latest/index.html) for details.

Use the environment variable `MLFLOW_TRACKING_URI` to set the URI of the MLFlow server. If not set the logging can not be used. An example of setting the URI to a server is and running a training command is `MLFLOW_TRACKING_URI=http://localhost:5000 python -m neural_lam.train_model --config_path <config_path> --logger mlflow`.

## Train Models
Models can be trained using `python -m neural_lam.train_model --config_path <config_path>`.
Run `python -m neural_lam.train_model --help` for a full list of training options.
A few of the key ones are outlined below:

* `--config_path`: Path to the configuration for neural-lam (for example in `data/myexperiment/config.yaml`).
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--epochs`: Number of epochs to train for
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps_train`: Number of time steps to unroll for when making predictions and computing the loss
* `--ar_steps_eval`: Number of time steps to unroll for during validation steps
* `--loss`: Which of the loss functions in `neural_lam/metrics.py` to train on

Checkpoints of trained models are stored under `runs/<run-name>/checkpoints/`, alongside Lightning and logger outputs for the same run.

### Transfer learning

Selected model components can initialize a fresh training run from a trusted
Lightning checkpoint. This workflow currently supports GraphLAM. A dotted
module path identifies a nested module by
joining its attribute names with periods. For example,
`forecaster.predictor.processor` means the `processor` module inside
`predictor`, which is inside `forecaster`. This command transfers GraphLAM's
processor and mesh embedders while leaving dataset-dependent input and output
layers newly initialized:

```bash
python -m neural_lam.train_model \
    --config_path <target-config> \
    --model graph_lam \
    --graph <target-graph> \
    --pretrained_checkpoint <source-checkpoint> \
    --pretrained_components \
        forecaster.predictor.processor \
        forecaster.predictor.mesh_embedder \
        forecaster.predictor.m2m_embedder
```

Add `--freeze_components forecaster.predictor.processor` to keep the
transferred processor parameters fixed during training. Freezing is a run
setting rather than checkpoint state, so repeat it when resuming a run that
should remain frozen.

`--pretrained_checkpoint` starts a fresh run with a new optimizer, epoch and
trainer state. It is therefore distinct from, and cannot be combined with,
`--load`, which resumes a complete training checkpoint. Every selected
component must exist in both models with exactly matching state keys and tensor
shapes. Selective transfer also requires the same normalized graph structure,
node ordering and graph features. Neural-LAM records graph identity before
runtime precision conversion, so training precision does not affect this check.
Matching graph identity does not verify dataset-variable meaning, ordering,
units, normalization statistics, or scientific suitability.

Known GraphLAM execution settings, including mesh aggregation, must also match.
Use `--allow_incompatible_pretrained_behavior` only for an intentional
behavior-changing experiment; it warns with the accepted differences. A graph
mismatch cannot be overridden, and cross-topology transfer is unsupported.
Checkpoints created before compatibility metadata was introduced are rejected by
default. To use a trusted legacy checkpoint after verifying its graph and model
settings, add `--allow_missing_compatibility_metadata`; this emits a warning and
makes the caller responsible for both forms of compatibility.

The implemented models are:

### Graph-LAM
This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step pedictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](#graph-based-neural-weather-prediction-for-limited-area-modeling), only with different graphs.

To train 1L-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph multiscale ...
```

### Hi-LAM
A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python -m neural_lam.train_model --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in parallel.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.
# From PyPI
python -m pip install neural_lam

# From source (recommended, using uv)
git clone https://github.com/mllam/neural-lam.git && cd neural-lam
uv sync --extra cpu --group dev   # CPU
uv sync --extra gpu --group dev   # GPU / CUDA 13.0
```

For all install options (pip, CUDA variants, editable installs) see the [Installation guide](https://mllam.github.io/neural-lam/getting-started/installation.html).

# Using Neural-LAM

You need a config file, a datastore, and a graph.
The [Quickstart](https://mllam.github.io/neural-lam/getting-started/quickstart.html) creates a graph, trains and evaluates a model from the command line on a small example dataset, and the [DANRA tutorial](https://mllam.github.io/neural-lam/notebooks/hello_world_danra.html) does the same in Python with plots.
The [user guide](https://mllam.github.io/neural-lam/user-guide/datastores.html) covers each step in detail, including how to set up a datastore for your own data, e.g. with [mllam-data-prep](https://github.com/mllam/mllam-data-prep).

# Repository Structure

```
neural-lam/
├── neural_lam/           - Source code
│   ├── train_model.py    - Training / evaluation CLI
│   ├── create_graph_with_wmg.py - Graph creation CLI
│   ├── api.py            - Python API (train, evaluate, create_graph)
│   ├── models/           - Forecasting modules, forecasters, step predictors and latent encoders/decoders (see below)
│   ├── datastore/        - Datastore implementations and base classes
│   └── utils/            - Graph loading and other helpers
├── docs/                 - Documentation source (Sphinx)
│   ├── getting-started/  - Installation and quickstart
│   ├── user-guide/       - Datastores, graphs, training and evaluation, logging
│   └── notebooks/        - Marimo tutorial notebooks
└── tests/                - Test suite and example datastore configs
```

The model code is split into three layers:

- `BaseForecastingModule` wraps a forecaster in the Lightning training/evaluation loop, handling optimization, logging, plotting and batch standardization.
  Its deterministic and probabilistic subclasses evaluate single forecasts and sampled ensembles, respectively.
- `BaseForecaster` maps initial states and forcing to a full forecast through `forward` and owns the training objective.
  `BaseDeterministicForecaster` returns the same forecast every call; `BaseProbabilisticForecaster` samples a fresh one, with `sample_ensemble` stacking repeated samples.
- `StepPredictor` advances the state one time step.
  The autoregressive forecasters (`DeterministicARForecaster` and `BaseProbabilisticARForecaster`) repeatedly apply it through the shared `unroll_forecast` function.

A forecaster is free to produce a forecast in any way, including predicting all lead times at once; only the autoregressive versions use unrolling.

All of these are `torch.nn.Module` subclasses, and `*` marks a class that is abstract and meant to be subclassed:
```
nn.Module
├── pl.LightningModule
│   └── BaseForecastingModule*             - Optimizer, logging, plotting and batch standardization (neural_lam/models/modules)
│       ├── DeterministicForecastingModule - Evaluates the single forecast the forecaster produces
│       └── ProbabilisticForecastingModule - Samples an ensemble and scores its mean
├── BaseForecaster*                        - Initial states + forcing -> full forecast; owns the training objective (neural_lam/models/forecasters)
│   ├── BaseDeterministicForecaster*       - forward -> the same forecast every call; objective: a loss function applied to it
│   │   └── DeterministicARForecaster      - Unrolls that forecast; what neural_lam.train_model builds
│   └── BaseProbabilisticForecaster*       - forward -> a fresh sample each call; sample_ensemble stacks several, leaving the objective open
│       └── BaseProbabilisticARForecaster* - Unrolls each sample, still without an objective
├── StepPredictor*                         - (X_{t-1}, X_t, forcing_t) -> X_{t+1} (neural_lam/models/step_predictors)
│   ├── BaseGraphModel*                    - Encode-process-decode over a mesh graph
│   │   ├── GraphLAM
│   │   └── BaseHiGraphModel*              - Requires a hierarchical mesh graph
│   │       ├── HiLAM
│   │       └── HiLAMParallel
│   └── BaseGraphEFM*                      - Latent-variable predictors with their own prior, encoder and decoder
│       ├── GraphEFM
│       └── GraphEFMMultiScale
├── BaseLatentEncoder*                     - Grid representation -> Gaussian over a latent on the mesh (neural_lam/models/latent)
│   ├── ConstantLatentEncoder
│   ├── GraphLatentEncoder
│   └── HiGraphLatentEncoder
└── BaseGraphLatentDecoder*                - Grid representation + latent sample -> next-state increment (neural_lam/models/latent)
    ├── GraphLatentDecoder
    └── HiGraphLatentDecoder
```
The `--model` values listed in the [training guide](https://mllam.github.io/neural-lam/user-guide/training.html) select the step predictor.
`GraphEFM` and `GraphEFMMultiScale` are implemented but not yet selectable this way.

The building blocks these are assembled from are also `torch.nn.Module` subclasses:
```
nn.Module
├── pyg.nn.MessagePassing
│   └── InteractionNet   - GNN layer, chosen per edge type with --g2m_gnn_type, --m2g_gnn_type, --mesh_up_gnn_type, --mesh_down_gnn_type (neural_lam/gnn_layers.py)
│       └── PropagationNet
├── BufferList           - List of tensor buffers held as a Module (neural_lam/utils/buffer_list.py)
└── SplitMLPs            - Feeds chunks of the input through separate MLPs (neural_lam/gnn_layers.py)
```

What to extend depends on the kind of model:

* **Deterministic, autoregressive:** write a new `StepPredictor` and run it with the existing `DeterministicARForecaster`.
  This is the common case.
* **Deterministic, not autoregressive:** subclass `BaseDeterministicForecaster`, implementing `forward` with the new way of producing a forecast, reusing an existing `StepPredictor` or a new one if the model has a per-step component at all.
* **Probabilistic, autoregressive:** subclass `BaseProbabilisticARForecaster`, implementing the training objective, typically alongside a new `StepPredictor` that samples its output.
* **Probabilistic, not autoregressive:** subclass `BaseProbabilisticForecaster`, which additionally leaves the way a single sample is produced open.

## Dimension Glossary

Canonical dimension names used in tensor shape annotations throughout the codebase:

- `B` - batch size
- `S` - number of members in an ensemble forecast
- `pred_steps` - number of autoregressive prediction steps
- `num_times` - number of time steps along the time axis of a raw or batched timeseries (a trailing `'`, e.g. `num_times'`, denotes a pre-subsampling / pre-differencing variant)
- `num_grid_nodes` - number of nodes in the flattened spatial grid
- `num_boundary_grid_nodes` - number of nodes in the flattened boundary spatial grid
- `num_mesh_nodes` - number of mesh nodes; indexed as `num_mesh_nodes[l]` for hierarchical level `l`
- `num_state_vars` - number of atmospheric state variables
- `num_forcing_vars` - number of forcing input variables
- `num_windowed_forcing_vars` - forcing variables stacked over the past/future forcing window
- `num_windowed_boundary_vars` - boundary forcing variables stacked over the past/future boundary window
- `num_variables` - generic variable dimension used in metric functions
- `hidden_dim` - internal hidden representation size in GNN layers and MLPs
- `latent_dim` - dimensionality of the Graph-EFM latent variable
- `input_dim` - input feature dimensionality to a layer before transformation
- `num_mesh_static_vars` - number of static features per mesh node
- `num_edges` - number of edges in a generic edge set, e.g. in message-passing layers
- `num_edges_<set>` - number of edges in a specific edge set, `<set>` being one of `g2m`, `m2g`, `m2m`, `mesh_up`, `mesh_down`; indexed as e.g. `num_edges_m2m[l]` for the per-level lists of a hierarchical graph
- `num_edge_features` - number of static features per edge
- `grid_size_x` - number of grid points along the x axis of a rectilinear grid
- `grid_size_y` - number of grid points along the y axis of a rectilinear grid
- `num_send` - number of sender nodes in a message-passing step
- `num_rec` - number of receiver nodes in a message-passing step

# Publications

*If you use Neural-LAM in your work, please cite the relevant paper(s).*

#### [Graph-based Neural Weather Prediction for Limited Area Modeling](https://arxiv.org/abs/2309.17370)
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
See branch [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) to reproduce this paper.

#### [Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks](https://arxiv.org/abs/2406.04759)
```
@inproceedings{oskarsson2024probabilistic,
  title = {Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
  author = {Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
}
```
See branches [`prob_model_lam`](https://github.com/mllam/neural-lam/tree/prob_model_lam) and [`prob_model_global`](https://github.com/mllam/neural-lam/tree/prob_model_global) to reproduce this paper.

# Development and Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

Join the [mllam Slack](https://kutt.to/mllam) or open a [GitHub issue](https://github.com/mllam/neural-lam/issues) to get in touch.
