# Neural-LAM

**Graph-based neural weather prediction for Limited Area Modeling**

Neural-LAM is a PyTorch and PyTorch Lightning framework for high-resolution weather prediction using Graph Neural Networks. It provides a modular approach to Limited Area Modeling, supporting multiple graph-based architectures such as GraphLAM, HiLAM, and HiLAMParallel to process and predict meteorological data efficiently.

```{admonition} Get Started in 5 Minutes
:class: tip
Install Neural-LAM and run your first prediction.
See the {doc}`Getting Started Guide <getting-started/installation>`.
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting-started/installation
:link-type: doc
Installation guide and quickstart tutorial to get you up and running.
:::


:::{grid-item-card} 🏗️ Architecture
:link: architecture/overview
:link-type: doc
Understand the data flow, model structure, and design decisions.
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc
Auto-generated reference for all modules, classes, and functions.
:::

::::

## Key Features

- **Modular design**: Swap datastores, models, and graph structures independently
- **Multiple model architectures**: GraphLAM (flat), HiLAM (hierarchical), HiLAMParallel (parallel hierarchical)
- **Flexible data handling**: Abstract datastore interface supporting zarr, numpy, and custom formats via mllam-data-prep
- **Production-ready**: PyTorch Lightning for training, W&B/MLflow logging, checkpoint management

## Publications

If you use Neural-LAM in your research, please cite the relevant papers:

**NeurIPS 2024 Paper** ([Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks](https://arxiv.org/abs/2406.04759))
```bibtex
@inproceedings{oskarsson2024probabilistic,
  title = {Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
  author = {Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
}
```

**NeurIPS 2023 Workshop Paper** ([Graph-based Neural Weather Prediction for Limited Area Modeling](https://arxiv.org/abs/2309.17370))
```bibtex
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```

## Quick Links

- [GitHub Repository](https://github.com/mllam/neural-lam)
- [Issue Tracker](https://github.com/mllam/neural-lam/issues)
- [MLLAM Community Slack](https://kutt.it/mllam)
