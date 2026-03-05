# Third-party
import pytest
import torch

# First-party
from neural_lam import metrics


@pytest.mark.parametrize(
    "metric_name, expected_name, expected_tensor",
    [
        (
            "mse",
            "rmse",
            torch.tensor([[2.0, 12.0], [4.0, 16.0]], dtype=torch.float32),
        ),
        (
            "mae",
            "mae",
            torch.tensor([[2.0, 36.0], [8.0, 64.0]], dtype=torch.float32),
        ),
        (
            "crps_gauss",
            "crps_gauss",
            torch.tensor([[2.0, 36.0], [8.0, 64.0]], dtype=torch.float32),
        ),
        (
            "wmse",
            "wrmse",
            torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32),
        ),
        (
            "wmae",
            "wmae",
            torch.tensor([[1.0, 9.0], [4.0, 16.0]], dtype=torch.float32),
        ),
        (
            "nll",
            "nll",
            torch.tensor([[1.0, 9.0], [4.0, 16.0]], dtype=torch.float32),
        ),
        (
            "spread_squared",
            "spread",
            torch.tensor([[2.0, 12.0], [4.0, 16.0]], dtype=torch.float32),
        ),
        (
            "output_std",
            "output_std",
            torch.tensor([[2.0, 36.0], [8.0, 64.0]], dtype=torch.float32),
        ),
    ],
)
def test_prepare_metric_tensor_for_logging(metric_name, expected_name, expected_tensor):
    metric_tensor = torch.tensor(
        [[1.0, 9.0], [4.0, 16.0]], dtype=torch.float32
    )
    state_std = torch.tensor([2.0, 4.0], dtype=torch.float32)

    metric_logged, log_name = metrics.prepare_metric_tensor_for_logging(
        metric_tensor=metric_tensor,
        metric_name=metric_name,
        state_std=state_std,
    )

    assert log_name == expected_name
    torch.testing.assert_close(metric_logged, expected_tensor)


def test_prepare_metric_tensor_for_logging_requires_explicit_spec():
    with pytest.raises(ValueError, match="Missing metric logging specification"):
        metrics.prepare_metric_tensor_for_logging(
            metric_tensor=torch.ones((1, 1), dtype=torch.float32),
            metric_name="unknown_metric",
            state_std=torch.ones(1, dtype=torch.float32),
        )


def test_crps_gauss_scales_linearly_with_data_scale():
    pred = torch.tensor([[[0.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0]]], dtype=torch.float32)
    pred_std = torch.tensor([[[2.0]]], dtype=torch.float32)

    base = metrics.crps_gauss(pred, target, pred_std)
    scaled = metrics.crps_gauss(10 * pred, 10 * target, 10 * pred_std)

    torch.testing.assert_close(scaled, 10 * base)


def test_nll_is_not_linearly_rescaled():
    pred = torch.tensor([[[0.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0]]], dtype=torch.float32)
    pred_std = torch.tensor([[[2.0]]], dtype=torch.float32)

    base = metrics.nll(pred, target, pred_std)
    scaled = metrics.nll(10 * pred, 10 * target, 10 * pred_std)

    assert not torch.isclose(scaled, 10 * base).item()
