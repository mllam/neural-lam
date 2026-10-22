---
name: add-metric
description: Implement, register, and test a new evaluation metric in Neural-LAM. Use when adding or modifying evaluation metrics or probabilistic scoring rules.
---

# Add an Evaluation Metric

Follow these instructions to implement, register, and test a new evaluation metric in Neural-LAM.

## 1. Locate Metric Definitions
All evaluation metrics and loss functions are defined and registered in `neural_lam/metrics.py`.

Key components:
- `DEFINED_METRICS`: Dictionary mapping lowercase metric names (e.g., `"mse"`, `"wmse"`, `"crps_gauss"`) to callable functions.
- `mask_and_reduce_metric`: Helper function that handles boolean masking over spatial grid points and tensor reduction (averaging over grid nodes, summing over state variables).
- `get_metric(metric_name)`: Factory returning the requested metric from `DEFINED_METRICS`.

## 2. Implement the Metric Function
Metric functions operate on predicted and ground-truth PyTorch tensors.

Signature pattern:
```python
import torch
from neural_lam.metrics import mask_and_reduce_metric


def my_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    """
    Calculate custom metric.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values of shape (..., num_grid_nodes, num_variables).
    target : torch.Tensor
        Ground truth values of shape (..., num_grid_nodes, num_variables).
    pred_std : torch.Tensor or None, optional
        Predicted standard deviation for probabilistic metrics.
    mask : torch.Tensor or None, optional
        Boolean spatial mask of shape (num_grid_nodes,).
    average_grid : bool, default True
        Whether to average across grid nodes.
    sum_vars : bool, default True
        Whether to sum across variable dimensions.

    Returns
    -------
    torch.Tensor
        Reduced metric tensor.
    """
    entry_diff = torch.abs(pred - target)
    return mask_and_reduce_metric(
        entry_diff,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )
```

## 3. Register in `DEFINED_METRICS`
Add the new function to `DEFINED_METRICS` in `neural_lam/metrics.py`:

```python
DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "my_metric": my_metric,
}
```

## 4. Add Unit Tests
Add unit tests in `tests/test_probabilistic_objectives.py` or a dedicated test file:

```python
import torch
from neural_lam.metrics import get_metric


def test_my_metric_registration_and_computation():
    metric_func = get_metric("my_metric")
    pred = torch.zeros(2, 10, 4)
    target = torch.ones(2, 10, 4)
    val = metric_func(pred, target)
    assert val.ndim == 1  # Reduced across grid and vars
    assert torch.isclose(val, torch.tensor([4.0, 4.0]))
```

## 5. Verification Check
Execute the metric test suite:

```bash
uv run pytest tests/test_probabilistic_objectives.py
```
