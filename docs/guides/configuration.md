# Configuration Reference

Neural-LAM uses `dataclass-wizard` to automatically parse and enforce type-safety for YAML configuration files. The configuration encapsulates both the **Datastore** configuration and the **Model/Training** configuration.

Below are common templates you can use for your own runs.

## 1. Quick CPU Test Run

This minimal configuration uses a dummy datastore, perfect for quickly testing changes on a laptop without a GPU.

```yaml
# tests/test_config.yaml
datastore:
  _target_: "neural_lam.datastore.DummyDatastore"

architecture: "graph_lam"
epochs: 2
batch_size: 2
lr: 1e-3
hidden_dim: 32
hidden_layers: 1
```

## 2. MDP (Meteorological Data Processing) Full Training

This is a production-level configuration for training on a real Zarr dataset produced by `mllam-data-prep` on a GPU cluster.

```yaml
# config/mdp_training.yaml
datastore:
  _target_: "neural_lam.datastore.MDPDatastore"
  dataset_path: "/path/to/my/zarr_dataset.zarr"
  subset_name: "meps" # Optional subset

# Model architecture settings
architecture: "hi_lam_parallel"
hidden_dim: 128
hidden_layers: 4
mesh_aggr: "sum"

# GNN Types for the different edges
g2m_gnn_type: "interaction"
m2g_gnn_type: "interaction"
mesh_up_gnn_type: "propagation"
mesh_down_gnn_type: "propagation"

# Training parameters
epochs: 100
batch_size: 16
lr: 5e-4
loss: "mse"

# Logging and reproducibility
seed: 42
```

## API Reference
For an exhaustive list of every configurable field and its default value, refer to the auto-generated documentation for the `NeuralLAMConfig` dataclass in the {py:mod}`neural_lam.config` module.
