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
# From PyPI
python -m pip install neural_lam

# From source (recommended, using uv)
git clone https://github.com/mllam/neural-lam.git && cd neural-lam
uv sync --extra cpu --group dev   # CPU
uv sync --extra gpu --group dev   # GPU / CUDA 13.0
```

For all install options (pip, CUDA variants, editable installs) see the [Installation guide](https://mllam.github.io/neural-lam/getting-started/installation.html).

# Using Neural-LAM

You need a config file, a datastore, and a graph. The fastest path:

```bash
# 1. Build the graph
python -m neural_lam.create_graph_with_wmg --config_path data/config.yaml --archetype keisler --name keisler

# 2. Train
python -m neural_lam.train_model --config_path data/config.yaml --model graph_lam --graph keisler

# 3. Evaluate
python -m neural_lam.train_model --eval test --config_path data/config.yaml --model graph_lam --graph keisler --load runs/<run-name>/checkpoints/min_val_loss.ckpt
```

The [DANRA tutorial](https://mllam.github.io/neural-lam/notebooks/hello_world_danra.html) runs these steps end to end on a small public example dataset. See the [configuration and datastores guide](https://mllam.github.io/neural-lam/user-guide/datastores.html) for how to write `config.yaml` and set up a datastore, e.g. with [mllam-data-prep](https://github.com/mllam/mllam-data-prep).

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
│   └── notebooks/        - Marimo tutorial notebooks
└── tests/                - Test suite and example datastore configs
```

## Dimension Glossary

Canonical dimension names used in tensor shape annotations throughout the codebase:

- `B` - batch size
- `pred_steps` - number of autoregressive prediction steps
- `num_grid_nodes` - number of nodes in the flattened spatial grid
- `num_mesh_nodes` - number of mesh nodes; indexed as `num_mesh_nodes[l]` for hierarchical level `l`
- `num_state_vars` - number of atmospheric state variables
- `num_forcing_vars` - number of forcing input variables
- `num_variables` - generic variable dimension used in metric functions
- `hidden_dim` - internal hidden representation size in GNN layers and MLPs
- `input_dim` - input feature dimensionality to a layer before transformation
- `num_edges` - number of edges in a graph (g2m, m2g, same-level, up, down)
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