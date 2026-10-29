"""Tests for gradient checkpointing during autoregressive rollout."""

# Standard library
from unittest.mock import patch

# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam.models.forecasters.autoregressive import ARForecaster
from neural_lam.models.step_predictors.base import StepPredictor
from tests.dummy_datastore import DummyDatastore


class TrainableStepPredictor(StepPredictor):
    """Minimal trainable step predictor for gradient checkpointing tests."""

    def __init__(self, datastore, **kwargs):
        super().__init__(datastore, **kwargs)
        num_state = datastore.get_num_data_vars(category="state")
        d_forcing = datastore.get_num_data_vars(category="forcing")
        input_dim = 2 * num_state + d_forcing
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_state),
        )
        if self.output_std:
            self.std_net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_state),
                nn.Softplus(),
            )

    def forward(self, prev_state, prev_prev_state, forcing):
        x = torch.cat([prev_state, prev_prev_state, forcing], dim=-1)
        delta = self.net(x)
        pred_state = prev_state + delta
        if self.output_std:
            pred_std = self.std_net(x)
        else:
            pred_std = None
        return pred_state, pred_std


@pytest.mark.parametrize("output_std", [False, True])
def test_ar_forecaster_gradient_checkpointing_numerical_parity(output_std):
    """Verify forward and backward match with and without checkpointing."""
    torch.manual_seed(42)
    datastore = DummyDatastore()

    predictor_baseline = TrainableStepPredictor(
        datastore, output_std=output_std
    )
    predictor_ckpt = TrainableStepPredictor(datastore, output_std=output_std)
    # Ensure both models start with identical weights
    predictor_ckpt.load_state_dict(predictor_baseline.state_dict())

    forecaster_baseline = ARForecaster(
        predictor_baseline, datastore, gradient_checkpointing=False
    )
    forecaster_ckpt = ARForecaster(
        predictor_ckpt, datastore, gradient_checkpointing=True
    )

    b_size = 2
    num_grid = predictor_baseline.num_grid_nodes
    num_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")
    pred_steps = 4

    init_states = torch.randn(b_size, 2, num_grid, num_state)
    forcing = torch.randn(b_size, pred_steps, num_grid, d_forcing)
    boundary = torch.randn(b_size, pred_steps, num_grid, num_state)

    # Forward + backward on baseline
    out_baseline, std_baseline = forecaster_baseline(
        init_states, forcing, boundary
    )
    loss_baseline = out_baseline.sum()
    if std_baseline is not None:
        loss_baseline = loss_baseline + std_baseline.sum()
    loss_baseline.backward()

    # Forward + backward with checkpointing
    out_ckpt, std_ckpt = forecaster_ckpt(init_states, forcing, boundary)
    loss_ckpt = out_ckpt.sum()
    if std_ckpt is not None:
        loss_ckpt = loss_ckpt + std_ckpt.sum()
    loss_ckpt.backward()

    # Assert forward outputs match
    assert torch.allclose(out_baseline, out_ckpt, atol=1e-6)
    if output_std:
        assert std_baseline is not None
        assert std_ckpt is not None
        assert torch.allclose(std_baseline, std_ckpt, atol=1e-6)
    else:
        assert std_baseline is None
        assert std_ckpt is None

    # Assert all gradients match
    for (name1, p1), (name2, p2) in zip(
        predictor_baseline.named_parameters(),
        predictor_ckpt.named_parameters(),
        strict=True,
    ):
        assert name1 == name2
        assert p1.grad is not None
        assert p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
            f"Gradients mismatch for parameter {name1}"
        )


@pytest.mark.parametrize(
    ("training", "checkpointing", "expected_calls"),
    [
        (True, True, 4),
        (True, False, 0),
        (False, True, 0),
        (False, False, 0),
    ],
)
def test_ar_forecaster_checkpoint_called_only_during_training(
    training, checkpointing, expected_calls
):
    """Verify checkpointing is invoked during training and skipped otherwise."""
    datastore = DummyDatastore()
    predictor = TrainableStepPredictor(datastore)
    forecaster = ARForecaster(
        predictor, datastore, gradient_checkpointing=checkpointing
    )
    forecaster.train(training)

    b_size = 2
    num_grid = predictor.num_grid_nodes
    num_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")
    pred_steps = 4

    init_states = torch.randn(b_size, 2, num_grid, num_state)
    forcing = torch.randn(b_size, pred_steps, num_grid, d_forcing)
    boundary = torch.randn(b_size, pred_steps, num_grid, num_state)

    with patch(
        "torch.utils.checkpoint.checkpoint",
        wraps=torch.utils.checkpoint.checkpoint,
    ) as mock_checkpoint:
        forecaster(init_states, forcing, boundary)
        assert mock_checkpoint.call_count == expected_calls
