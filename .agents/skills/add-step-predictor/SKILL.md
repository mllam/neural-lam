---
name: add-step-predictor
description: Implement, register, and test a single-step prediction model using StepPredictor or BaseGraphModel.
---

# Add a Step Predictor Model

Follow this recipe to implement a single-step prediction model, register it in `neural_lam.models.MODELS`, and test it against `DummyDatastore`.

## 1. Subclass `StepPredictor` or `BaseGraphModel`
Implement your model in `neural_lam/models/step_predictors/<model_name>.py`:
- For general single-step predictors, inherit from `neural_lam.models.step_predictors.base.StepPredictor`.
- For graph-based mesh models, inherit from `neural_lam.models.step_predictors.graph.base.BaseGraphModel`.

### Conventions & Interface Contract
- **Constructor**: `__init__(self, datastore, **kwargs)` must accept the datastore instance and absorb CLI / configuration kwargs (e.g. `output_std`, `hidden_dim`, `processor_layers`).
- **Forward Signature**:
```python
def forward(
    self,
    prev_state: torch.Tensor,
    prev_prev_state: torch.Tensor,
    forcing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ...
```
  - `prev_state`: Current state $X_t$ of shape `(B, num_grid_nodes, num_state_vars)`.
  - `prev_prev_state`: Prior state $X_{t-1}$ of shape `(B, num_grid_nodes, num_state_vars)`.
  - `forcing`: External forcing features of shape `(B, num_grid_nodes, num_forcing_vars)`.
  - Return: 2-tuple `(pred_state, pred_std)` where `pred_std` is `None` for deterministic predictions.

### Reference Implementations
Inspect these minimal implementations as canonical references:
- **Smallest minimal implementation**: `neural_lam.models.step_predictors.persistence.PersistencePredictor` in `neural_lam/models/step_predictors/persistence.py`.
- **Graph model implementation**: `neural_lam.models.step_predictors.graph.graph_lam.GraphLAM` in `neural_lam/models/step_predictors/graph/graph_lam.py`.

## 2. Register in `MODELS`
Register the model class in `neural_lam.models.MODELS` in `neural_lam/models/__init__.py` with its CLI identifier string.

## 3. Extend Unit Tests
Add test cases verifying model instantiation, forward shapes, and device execution by extending:
- `tests/test_prediction_model_classes.py`
- Or a dedicated test module following `tests/test_persistence.py`.

Use `tests.dummy_datastore.DummyDatastore` for fast in-memory testing without downloading remote datasets.

## 4. Verification Check
Run model tests:
```bash
uv run pytest tests/test_prediction_model_classes.py
```
