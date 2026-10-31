---
name: add-metric
description: Implement, register, and test a new evaluation metric using DEFINED_METRICS and mask_and_reduce_metric.
---

# Add an Evaluation Metric

Follow this recipe to implement, register, and test a new evaluation metric in Neural-LAM.

## 1. Implement the Metric Function
Add your metric function in `neural_lam/metrics.py`:
- Use `mask_and_reduce_metric` to apply spatial masking and tensor reductions across grid and variable dimensions.

### Conventions & Interface Contract
Metric functions in `neural_lam.metrics` follow the standard callable contract:
```python
def my_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    ...
```
- `pred`: Model prediction tensor of shape `(..., N, num_variables)`.
- `target`: Ground truth target tensor of shape `(..., N, num_variables)`.
- `pred_std`: Predicted standard deviation or weighting tensor (can be ignored or replaced with ones for deterministic metrics).
- `mask`: Optional boolean spatial mask of shape `(N,)` over active grid nodes.
- `average_grid`: If `True`, average across the grid dimension `N`.
- `sum_vars`: If `True`, sum across the variable dimension `num_variables`.

### Reference Implementations
Inspect these implementations in `neural_lam/metrics.py`:
- **Smallest deterministic reference**: `mae` in `neural_lam/metrics.py`.
- **Weighted baseline reference**: `wmse` in `neural_lam/metrics.py`.
- **Spatial reduction utility**: `mask_and_reduce_metric` in `neural_lam/metrics.py`.

## 2. Register in `DEFINED_METRICS`
Register the metric function in `neural_lam.metrics.DEFINED_METRICS` in `neural_lam/metrics.py`:
- Add a lowercase string key mapping to your callable (e.g. `"my_metric": my_metric`).
- Accessible via `neural_lam.metrics.get_metric("my_metric")`.

## 3. Extend Unit Tests
Add verification tests in `tests/test_probabilistic_objectives.py`:
- Test metric registration via `get_metric("my_metric")`.
- Test known analytical outputs with synthetic zero and constant tensors.
- Test spatial masking behavior (`mask=...`).

## 4. Verification Check
Run metric tests:
```bash
uv run pytest tests/test_probabilistic_objectives.py
```
