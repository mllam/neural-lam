---
name: add-step-predictor
description: Implement, register, and test a new single-step prediction architecture in Neural-LAM. Use when creating new neural network backbones or graph prediction models.
---

# Add a Step Predictor Model

Follow these instructions to implement a new single-step weather prediction model, register it in `neural_lam.models.MODELS`, and test it with `DummyDatastore`.

## 1. Subclass `StepPredictor` or `BaseGraphModel`
All step predictors in Neural-LAM inherit from `StepPredictor` (in `neural_lam/models/step_predictors/base.py`) or `BaseGraphModel` (in `neural_lam/models/step_predictors/graph/base.py`).

Implement your model in `neural_lam/models/step_predictors/<model_name>.py`:

```python
import torch
import torch.nn as nn
from neural_lam.models.step_predictors.base import StepPredictor


class MyCustomPredictor(StepPredictor):
    """Custom neural step predictor."""

    def __init__(
        self,
        datastore,
        hidden_dim: int = 64,
        output_std: bool = False,
        **kwargs,
    ):
        super().__init__(datastore=datastore, output_std=output_std, **kwargs)
        self.hidden_dim = hidden_dim

        # Input dimension: 2 * state vars + forcing vars
        in_dim = 2 * self.d_state + self.d_forcing
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.d_state),
        )

        if self.output_std:
            self.std_head = nn.Sequential(
                nn.Linear(hidden_dim, self.d_state),
                nn.Softplus(),
            )

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Predict next atmospheric state.

        Parameters
        ----------
        prev_state : torch.Tensor
            State at t-1 of shape (B, num_grid_nodes, d_state).
        prev_prev_state : torch.Tensor
            State at t-2 of shape (B, num_grid_nodes, d_state).
        forcing : torch.Tensor
            External forcing features of shape (B, num_grid_nodes, d_forcing).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Predicted state difference/value and optional predicted standard deviation.
        """
        inputs = torch.cat([prev_state, prev_prev_state, forcing], dim=-1)
        pred_delta = self.net(inputs)
        pred_state = prev_state + pred_delta

        pred_std = None
        if self.output_std:
            pred_std = self.std_head(self.net[0](inputs))

        return pred_state, pred_std
```

## 2. Register in `neural_lam.models.MODELS`
Register the class in `neural_lam/models/__init__.py`:

```python
from .step_predictors.my_model import MyCustomPredictor

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
    "persistence": PersistencePredictor,
    "my_custom_model": MyCustomPredictor,
}
```

## 3. Test Integration
Add unit tests in `tests/test_prediction_model_classes.py` verifying model instantiation and forward shape assertions:

```python
import torch
from neural_lam.models import MODELS
from tests.dummy_datastore import DummyDatastore


def test_my_custom_model_forward():
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
    model_cls = MODELS["my_custom_model"]
    model = model_cls(datastore=datastore, hidden_dim=16)

    B, N, d_state = 2, datastore.grid_shape_state.n_points, 5
    d_forcing = 2
    prev_state = torch.zeros(B, N, d_state)
    prev_prev_state = torch.zeros(B, N, d_state)
    forcing = torch.zeros(B, N, d_forcing)

    pred, pred_std = model(prev_state, prev_prev_state, forcing)
    assert pred.shape == (B, N, d_state)
    assert pred_std is None
```

## 4. Verification Check
Run model unit tests:

```bash
uv run pytest tests/test_prediction_model_classes.py -k my_custom_model
```

Run a 1-epoch offline training test using the programmatic API:

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
run = api.train(
    datastore=datastore,
    config=config,
    model="my_custom_model",
    graph="1level",
    epochs=1,
    batch_size=1,
    num_workers=0,
)
```
