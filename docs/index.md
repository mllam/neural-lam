# Neural-LAM

Neural-LAM is a PyTorch and PyTorch Lightning framework for high-resolution weather prediction using Graph Neural Networks. It provides a modular approach to Limited Area Modeling, supporting multiple graph-based architectures such as GraphLAM, HiLAM, and HiLAMParallel to process and predict meteorological data efficiently.

```{toctree}
:maxdepth: 2
:hidden:

getting-started/installation
getting-started/quickstart
notebooks/create_reduced_meps_dataset
api/index
```

- **[🚀 Getting Started](getting-started/installation.md)**: Installation guide and quickstart tutorial to get you up and running.
- **[📚 API Reference](api/index)**: Auto-generated reference for all modules, classes, and functions.

## Key Features

- **Modular design**: Swap datastores, models, and graph structures independently
- **Multiple model architectures**: GraphLAM (flat), HiLAM (hierarchical), HiLAMParallel (parallel hierarchical)
- **Flexible data handling**: Abstract datastore interface supporting zarr, numpy, and custom formats via mllam-data-prep
- **Production-ready**: PyTorch Lightning for training, W&B/MLflow logging, checkpoint management

## Quick Links

- [GitHub Repository](https://github.com/mllam/neural-lam)
- [Issue Tracker](https://github.com/mllam/neural-lam/issues)
- [MLLAM Community Slack](https://kutt.to/mllam)
