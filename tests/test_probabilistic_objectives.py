# Third-party
import torch

# First-party
from neural_lam.metrics import crps_gauss, nll, wmae, wmse


def _single_residual_case(residual=1.0):
    """Create a minimal broadcast-friendly prediction/target pair."""
    pred = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    target = torch.full_like(pred, residual)
    return pred, target


def test_wmse_rewards_variance_inflation_for_fixed_residual():
    pred, target = _single_residual_case()

    base_std = torch.tensor([1.0], dtype=torch.float32)
    doubled_std = torch.tensor([2.0], dtype=torch.float32)

    base_loss = wmse(pred, target, base_std)
    doubled_loss = wmse(pred, target, doubled_std)

    torch.testing.assert_close(doubled_loss, base_loss / 4)


def test_wmae_rewards_variance_inflation_for_fixed_residual():
    pred, target = _single_residual_case()

    base_std = torch.tensor([1.0], dtype=torch.float32)
    doubled_std = torch.tensor([2.0], dtype=torch.float32)

    base_loss = wmae(pred, target, base_std)
    doubled_loss = wmae(pred, target, doubled_std)

    torch.testing.assert_close(doubled_loss, base_loss / 2)


def test_nll_prefers_calibrated_std_to_extreme_scales():
    pred, target = _single_residual_case()

    too_small_std = torch.tensor([0.2], dtype=torch.float32)
    residual_scale_std = torch.tensor([1.0], dtype=torch.float32)
    too_large_std = torch.tensor([5.0], dtype=torch.float32)

    small_loss = nll(pred, target, too_small_std)
    residual_scale_loss = nll(pred, target, residual_scale_std)
    large_loss = nll(pred, target, too_large_std)

    assert torch.all(residual_scale_loss < small_loss)
    assert torch.all(residual_scale_loss < large_loss)


def test_crps_gauss_prefers_calibrated_std_to_extreme_scales():
    pred, target = _single_residual_case()

    too_small_std = torch.tensor([0.2], dtype=torch.float32)
    residual_scale_std = torch.tensor([1.0], dtype=torch.float32)
    too_large_std = torch.tensor([5.0], dtype=torch.float32)

    small_loss = crps_gauss(pred, target, too_small_std)
    residual_scale_loss = crps_gauss(pred, target, residual_scale_std)
    large_loss = crps_gauss(pred, target, too_large_std)

    assert torch.all(residual_scale_loss < small_loss)
    assert torch.all(residual_scale_loss < large_loss)


def test_probabilistic_losses_support_pred_std_broadcasting():
    pred = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[[1.0, 0.0], [1.0, 5.0]]]],
        dtype=torch.float32,
    )
    pred_std_broadcast = torch.tensor([1.5, 0.5], dtype=torch.float32)
    pred_std_full = pred_std_broadcast.view(1, 1, 1, 2).expand_as(pred)

    torch.testing.assert_close(
        nll(
            pred,
            target,
            pred_std_broadcast,
            average_grid=False,
            sum_vars=False,
        ),
        nll(
            pred,
            target,
            pred_std_full,
            average_grid=False,
            sum_vars=False,
        ),
    )
    torch.testing.assert_close(
        crps_gauss(
            pred,
            target,
            pred_std_broadcast,
            average_grid=False,
            sum_vars=False,
        ),
        crps_gauss(
            pred,
            target,
            pred_std_full,
            average_grid=False,
            sum_vars=False,
        ),
    )
