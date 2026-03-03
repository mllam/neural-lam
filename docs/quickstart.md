# Quickstart Guide

This guide demonstrates how to use neural-lam for computing evaluation metrics.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mllam/neural-lam.git
cd neural-lam
pip install -e .
```

## Minimal Example

```python
import torch
from neural_lam.metrics import mse, get_metric

# Dummy predictions
pred = torch.randn(2, 10, 3)
target = torch.randn(2, 10, 3)
pred_std = torch.ones_like(pred)

# Compute MSE
loss = mse(pred, target, pred_std)
print("MSE:", loss)
```

## Using Registered Metrics

You can also retrieve metrics dynamically:

```python
metric_fn = get_metric("wmse")
value = metric_fn(pred, target, pred_std)
print("Weighted MSE:", value)
```

## Available Metrics

Currently registered metrics:

- mse
- wmse
- mae
- wmae
- nll
- crps_gauss