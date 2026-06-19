# Model Architectures

## Overview

Neural-LAM employs a modular model hierarchy based on the **encode-process-decode** paradigm. This paradigm allows the models to encode grid-based weather data into a graph structure, perform spatial and temporal message passing to process the data, and finally decode the updated graph states back into the original grid representation.

## Autoregressive Framework

The core training and evaluation loop is handled by `ARModel`, a PyTorch Lightning module defined in `neural_lam/models/module.py`. This autoregressive framework is responsible for unrolling predictions over multiple timesteps.

```{mermaid}
sequenceDiagram
    participant Trainer as Lightning Trainer
    participant AR as ARModel
    participant F as Forecaster
    participant SP as StepPredictor
    
    Trainer->>AR: training_step(batch)
    loop for each timestep
        AR->>F: forward(prev_state)
        F->>SP: predict_step(state, graph)
        SP-->>F: next_state_delta
        F-->>AR: next_state
    end
    AR->>AR: compute_loss()
    AR-->>Trainer: loss
```

## Encode-Process-Decode

The base graph model, defined in `neural_lam/models/step_predictors/base.py`, implements the standard encode-process-decode steps:
1. **Encode**: Features from the weather grid are mapped onto the nodes and edges of the mesh graph.
2. **Process**: A Graph Neural Network (GNN) performs multiple rounds of message passing to propagate information across the spatial domain.
3. **Decode**: The updated mesh features are mapped back to the grid to produce the next state prediction.

## GraphLAM

GraphLAM is the fundamental model architecture utilizing a single-level flat graph. It is effective for standard resolution forecasting without multi-scale processing.

Reference: {py:class}`neural_lam.models.step_predictors.graph.GraphLAM`

## HiLAM

HiLAM introduces a hierarchical model design, utilizing multiple levels of mesh nodes at increasing spatial scales. This allows the network to efficiently capture both local, fine-grained interactions and long-range, global atmospheric patterns.

Reference: {py:class}`neural_lam.models.step_predictors.graph.HiLAM`

## HiLAMParallel

HiLAMParallel is a parallel hierarchical variant of HiLAM. It processes multi-scale information simultaneously across different hierarchy levels, rather than sequentially, potentially improving computational efficiency and long-range interaction modeling.

Reference: {py:class}`neural_lam.models.step_predictors.graph.HiLAMParallel`

## Choosing a Model

| Model | Graph Type | Complexity | Best For |
|---|---|---|---|
| **GraphLAM** | Flat Mesh | Low | Baseline forecasting, single-scale dynamics |
| **HiLAM** | Hierarchical Mesh | Medium | Capturing both local and global dependencies efficiently |
| **HiLAMParallel** | Hierarchical Mesh | High | Highly parallel environments, very large spatial domains |
