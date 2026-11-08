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

## Step 2: Create a Graph

Before training, you must construct a graph mesh for your data.

```{code-block} bash
python -m neural_lam.create_graph_with_wmg --config_path <path-to-config> --name <graph-name>
```

This script builds the mesh graph required by the GNN models.

## Step 3: Train a Model

Now you can train a model using the graph and configuration.

```{code-block} bash
python -m neural_lam.train_model --config_path <path-to-config> --model graph_lam --graph <graph-name>
```

Neural-LAM supports several models like `graph_lam`, `hi_lam`, and `hi_lam_parallel`.

To start fresh training from selected components of a trusted checkpoint, use
`--pretrained_checkpoint` together with one or more dotted module paths:

```{code-block} bash
python -m neural_lam.train_model --config_path <path-to-config> --model graph_lam --graph <graph-name> --pretrained_checkpoint <checkpoint-path> --pretrained_components forecaster.predictor.processor
```

This differs from `--load`, which resumes the complete training state. Selected
components must have matching state keys and tensor shapes in the source and
target models. Graph-model checkpoints must also contain the graph fingerprint
used to reject incompatible topology, node ordering and graph features. A
trusted older checkpoint can be used with
`--allow_missing_compatibility_metadata` after manually verifying its graph;
known fingerprint mismatches are always rejected.

## Step 4: Evaluate

After training, you can evaluate the model on the test set by loading the saved checkpoint.

```{code-block} bash
python -m neural_lam.train_model --eval test --config_path <path-to-config> --model graph_lam --graph <graph-name> --load <checkpoint-path>
```

## Configuration

Neural-LAM uses a YAML configuration system powered by `dataclass-wizard`. The configuration selects the datastore and sets training options such as state feature weighting and output clamping; model hyperparameters are `train_model` CLI flags. For complete details, see the API reference for {py:class}`neural_lam.config.NeuralLAMConfig`.

- {doc}`../api/index` for complete API reference
