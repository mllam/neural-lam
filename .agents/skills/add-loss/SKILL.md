---
name: add-loss
description: Implement, configure, and test a training loss function with spatial masking and state-feature weighting.
---

# Add a Loss Function

Follow this recipe to implement and register a training loss function with spatial masking and state-feature weighting.

## 1. Implement Loss in `metrics.py`
In Neural-LAM, training losses share the standard callable metric interface and are resolved via `neural_lam.metrics.get_metric(loss_name)` inside `ForecasterModule`.
- Implement your loss function in `neural_lam/metrics.py`.
- Use `mask_and_reduce_metric` to apply spatial masking and tensor reductions across grid and variable dimensions.

### Conventions & Interface Contract
Loss functions follow the standard signature:
```python
def my_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    average_grid: bool = True,
    sum_vars: bool = True,
) -> torch.Tensor:
    ...
```
- `pred`: Predictions of shape `(..., N, num_variables)`.
- `target`: Ground truth targets of shape `(..., N, num_variables)`.
- `pred_std`: Predicted scale/variance or feature weights of shape `(..., N, num_variables)`.
- `mask`: Spatial mask of shape `(N,)` isolating interior grid nodes from boundary nodes.

### Reference Implementations & Weighting Pointers
Inspect these implementations as canonical references:
- **Weighted MSE loss reference**: `wmse` in `neural_lam/metrics.py`.
- **Weighted MAE loss reference**: `wmae` in `neural_lam/metrics.py`.
- **State-feature weighting utilities**: `neural_lam.loss_weighting.get_uniform_state_feature_weights` and `neural_lam.loss_weighting.get_manual_state_feature_weights` in `neural_lam/loss_weighting.py`.
- **Forecaster integration**: `neural_lam.models.ForecasterModule` in `neural_lam/models/module.py`.

## 2. Register in `DEFINED_METRICS`
Add the loss key to `DEFINED_METRICS` in `neural_lam/metrics.py` so it can be passed via `--loss <name>` or `api.train(loss="<name>")`.

## 3. Extend Unit Tests
Add test cases in `tests/test_probabilistic_objectives.py`:
- Test loss computation on known inputs.
- Test spatial boundary masking behavior.
- Test gradient backpropagation through the loss function.

## 4. Verification Check
Run loss unit tests:
```bash
uv run pytest tests/test_probabilistic_objectives.py
```
