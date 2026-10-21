[![slack](https://img.shields.io/badge/slack-join-brightgreen.svg?logo=slack)](https://kutt.to/mllam)
[![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml)
[![CPU+GPU testing](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml)

<p align="middle">
    <img src="https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
Also global forecasting is possible, but currently on a [different branch](https://github.com/mllam/neural-lam/tree/prob_model_global) ([planned to be merged with main](https://github.com/mllam/neural-lam/issues/63)).
The code uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).
Graph Neural Networks are implemented using [PyG](https://pyg.org/) and logging is set up through [Weights & Biases](https://wandb.ai/).

## Documentation

Full guides, tutorials, and API reference live at **[mllam.github.io/neural-lam](https://mllam.github.io/neural-lam/)**:

- 📦 [Installation](https://mllam.github.io/neural-lam/getting-started/installation.html)
- 🚀 [Quickstart](https://mllam.github.io/neural-lam/getting-started/quickstart.html) and the [DANRA tutorial](https://mllam.github.io/neural-lam/notebooks/hello_world_danra.html)
- 📖 User guide: [configuration and datastores](https://mllam.github.io/neural-lam/user-guide/datastores.html), [graph creation](https://mllam.github.io/neural-lam/user-guide/graphs.html), [training and evaluation](https://mllam.github.io/neural-lam/user-guide/training.html), [experiment logging](https://mllam.github.io/neural-lam/user-guide/logging.html)
- 📚 [API Reference](https://mllam.github.io/neural-lam/api/index.html)
- 🗺️ [Graph Storage Specification](https://mllam.github.io/neural-lam/graph_storage_spec.html)

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

# Publications
For a more in-depth scientific introduction to machine learning for LAM weather forecasting see the publications listed here.
As the code in the repository is continuously evolving, the latest version might feature some small differences to what was used for these publications.
We retain some paper-specific branches for reproducibility purposes.

### Core Neural-LAM Publications
*If you use Neural-LAM in your work, please cite the relevant paper(s)*.

#### [Graph-based Neural Weather Prediction for Limited Area Modeling](https://arxiv.org/abs/2309.17370)
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
See the branch [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) for a revision of the code that reproduces this workshop paper.

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
See the branches [`prob_model_lam`](https://github.com/mllam/neural-lam/tree/prob_model_lam) and [`prob_model_global`](https://github.com/mllam/neural-lam/tree/prob_model_global) for revisions of the code that reproduces this paper.
The global and probabilistic models from this paper are not yet fully merged with `main` (see issues [62](https://github.com/mllam/neural-lam/issues/62) and [63](https://github.com/mllam/neural-lam/issues/63)).

# Modularity
The Neural-LAM code is designed to modularize the different components involved in training and evaluating neural weather prediction models.
Models, graphs and data are stored separately and it should be possible to swap out individual components.
Still, some restrictions are inevitable:

* The graph used has to be compatible with what the model expects. E.g. a hierarchical model requires a hierarchical graph.
* The graph and data are specific to the limited area under consideration. This is of course true for the data, but also the graph should be created with the exact geometry of the area in mind.

<p align="middle">
  <img src="https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_setup.png" width="600"/>
</p>

# Installing Neural-LAM

See the [Installation guide](https://mllam.github.io/neural-lam/getting-started/installation.html) for full instructions including all pip/CUDA variants.

```bash
# From PyPI
python -m pip install neural_lam

# From source (recommended, using uv)
git clone https://github.com/mllam/neural-lam.git && cd neural-lam
uv sync --extra cpu --group dev   # CPU
uv sync --extra gpu --group dev   # GPU / CUDA 13.0
```

# Using Neural-LAM

You need a config file, a datastore, and a graph. The [Quickstart](https://mllam.github.io/neural-lam/getting-started/quickstart.html) creates a graph, trains and evaluates a model from the command line on a small example dataset, and the [DANRA tutorial](https://mllam.github.io/neural-lam/notebooks/hello_world_danra.html) does the same in Python with plots. The [user guide](https://mllam.github.io/neural-lam/user-guide/datastores.html) covers each step in detail, including how to set up a datastore for your own data, e.g. with [mllam-data-prep](https://github.com/mllam/mllam-data-prep).

# Repository Structure

```
neural-lam/
├── neural_lam/           - Source code
│   ├── train_model.py    - Training / evaluation CLI
│   ├── create_graph_with_wmg.py - Graph creation CLI
│   ├── api.py            - Python API (train, evaluate, create_graph)
│   ├── models/           - ForecasterModule, forecasters, step predictors (GraphLAM, HiLAM, HiLAMParallel, Persistence)
│   ├── datastore/        - Datastore implementations and base classes
│   └── utils/            - Graph loading and other helpers
├── docs/                 - Documentation source (Sphinx)
│   ├── getting-started/  - Installation and quickstart
│   ├── user-guide/       - Datastores, graphs, training and evaluation, logging
│   └── notebooks/        - Marimo tutorial notebooks
└── tests/                - Test suite and example datastore configs
```

## Dimension Glossary

Canonical dimension names used in tensor shape annotations throughout the codebase:

- `B` - batch size
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
- `input_dim` - input feature dimensionality to a layer before transformation
- `num_mesh_static_vars` - number of static features per mesh node
- `num_edges` - number of edges in a graph (g2m, m2g, same-level, up, down)
- `num_send` - number of sender nodes in a message-passing step
- `num_rec` - number of receiver nodes in a message-passing step

# Development and Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide: how to open an issue, set up a development environment, the code standards we follow, the checks to run before pushing, the pull-request and review process, and the community roadmap.

# Contact
If you are interested in machine learning models for LAM, have questions about the implementation or ideas for extending it, feel free to get in touch.
There is an open [mllam slack channel](https://kutt.to/mllam) that anyone can join (after following the link you have to request to join, this is to avoid spam bots).
You can also open a github issue on this page.
