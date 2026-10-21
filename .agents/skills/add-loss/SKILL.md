---
name: add-loss
description: Implement, configure, and test a loss function in Neural-LAM, including spatial masking and feature weighting. Use when creating custom training objectives.
---

# Add a Loss Function

Follow these instructions to implement and register a custom loss function with spatial masking and state-feature weighting.

## 1. Overview
In Neural-LAM, loss functions are resolved via `neural_lam.metrics.get_metric(loss_name)` inside `ForecasterModule`.
Loss computation receives prediction tensors, targets, optional predicted standard deviation, and an optional spatial mask.

Feature weighting can be configured uniformly or manually via `neural_lam.loss_weighting` and `neural_lam.config.StateFeatureWeighting`.

## 2. Implement the Loss Function
Implement the loss function in `neural_lam/metrics.py` (or a dedicated submodule imported into `metrics.py`):

```python
import torch
from neural_lam.metrics import mask_and_reduce_metric


def log_spectral_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Calculate penalty between pred and target with optional spatial masking.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values of shape (..., num_grid_nodes, num_variables).
    target : torch.Tensor
        Target values of shape (..., num_grid_nodes, num_variables).
    pred_std : torch.Tensor or None, optional
        Predicted standard deviation for probabilistic metrics.
    mask : torch.Tensor or None, optional
        Boolean spatial mask over grid nodes of shape (num_grid_nodes,).
    average_grid : bool, default True
        Whether to average across grid nodes.
    sum_vars : bool, default True
        Whether to sum across variable dimensions.

    Returns
    -------
    torch.Tensor
        Reduced loss tensor.
    """
    # Compute squared or custom difference
    diff_sq = (pred - target) ** 2

    # Reduce across spatial and variable dimensions
    return mask_and_reduce_metric(
        diff_sq,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )
```

## 3. Register in `DEFINED_METRICS`
Add the loss function key to `DEFINED_METRICS` in `neural_lam/metrics.py`:

```python
DEFINED_METRICS = {
    # Existing metrics
    "wmse": wmse,
    "wmae": wmae,
    "lsd": log_spectral_distance_loss,
}
```

## 4. Integrate Feature Weighting
If your loss supports state-feature weighting, retrieve the weights using `neural_lam.loss_weighting`:

```python
from neural_lam.loss_weighting import (
    get_manual_state_feature_weights,
    get_uniform_state_feature_weights,
)

# Feature weights align with state variables ordered by datastore
weights = get_uniform_state_feature_weights(datastore)
```

## 5. Verification Check
Test the loss calculation in `tests/test_probabilistic_objectives.py`:

```bash
uv run pytest tests/test_probabilistic_objectives.py
```

Test offline training for 1 epoch using the programmatic API with the new loss:

```python
from neural_lam import api
from neural_lam.config import DatastoreSelection, NeuralLAMConfig
from tests.dummy_datastore import DummyDatastore

datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
config = NeuralLAMConfig(
    datastores={
        "main": DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    }
)
api.create_graph(datastore=datastore, name="1level", archetype="keisler")
train_run = api.train(
    datastore=datastore,
    config=config,
    loss="lsd",
    epochs=1,
    batch_size=1,
    num_workers=0,
)
```
