# Quickstart

## Overview

This guide walks through the minimum steps to get started with Neural-LAM:
1. Set up example data
2. Create a graph
3. Train a model
4. Evaluate the model

## Step 1: Download Example Data

When running tests for the first time, example data is automatically downloaded from S3. You can trigger this download by running a minimal test.

```{code-block} bash
# Run a minimal test to trigger the data download
pytest tests/test_training.py -vv -s -k "test_training"
```

Alternatively, you can use the `DummyDatastore` for quick testing without downloading real data.

## Step 2: Create a Graph

Before training, you must construct a graph mesh for your data.

```{code-block} bash
python -m neural_lam.create_graph --config_path <path-to-config> --name <graph-name>
```

This script builds the mesh graph required by the GNN models.

## Step 3: Train a Model

Now you can train a model using the graph and configuration.

```{code-block} bash
python -m neural_lam.train_model --config_path <path-to-config> --model graph_lam --graph <graph-name>
```

Neural-LAM supports several models like `graph_lam`, `hilam`, and `hilam_parallel`.

## Step 4: Evaluate

After training, you can evaluate the model on the test set by loading the saved checkpoint.

```{code-block} bash
python -m neural_lam.train_model --eval test --config_path <path-to-config> --load <checkpoint-path>
```

## Configuration

Neural-LAM uses a YAML configuration system powered by `dataclass-wizard`. The configuration defines the dataset paths, training parameters, and model hyperparameters. For complete details, see the API reference for {py:class}`neural_lam.config.NeuralLAMConfig`.

- {doc}`../api/index` for complete API reference
