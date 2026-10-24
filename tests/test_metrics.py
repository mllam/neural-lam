# Third-party
import torch

# First-party
from neural_lam.metrics import bias, get_metric


def test_bias_is_signed_mean_error():
    pred = torch.tensor([[[2.0, 1.0], [4.0, 1.0]]])
    target = torch.tensor([[[1.0, 3.0], [1.0, 3.0]]])
    pred_std = torch.ones_like(pred)

    result = bias(pred, target, pred_std, average_grid=False, sum_vars=False)

    torch.testing.assert_close(result, pred - target)


def test_bias_reduction_matches_mean_over_grid_and_sum_over_vars():
    pred = torch.tensor([[[2.0, 1.0], [4.0, 1.0]]])
    target = torch.tensor([[[1.0, 3.0], [0.0, 3.0]]])
    pred_std = torch.ones_like(pred)

    result = bias(pred, target, pred_std)

    expected = torch.mean(pred - target, dim=-2).sum(dim=-1)
    torch.testing.assert_close(result, expected)


def test_bias_registered_as_defined_metric():
    assert get_metric("bias") is bias
