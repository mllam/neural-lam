# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam.utils import inverse_softplus


@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_inverse_softplus_roundtrip(beta):
    """`inverse_softplus` recovers the input of `softplus`."""
    # Range chosen so softplus(x, beta) stays above the lower clamp
    # log(1+1e-6)/beta for all tested betas (clamp activates around
    # x = -13.8/beta).
    x_orig = torch.linspace(-5, 5, steps=100)
    y = F.softplus(x_orig, beta=beta)
    x_reconstructed = inverse_softplus(y, beta=beta)
    torch.testing.assert_close(x_orig, x_reconstructed)


def test_inverse_softplus_near_zero_is_finite():
    """Near-zero inputs are clamped so the log path cannot produce NaN/-Inf."""
    y_near_zero = torch.tensor([1e-7, 1e-6])
    x_near_zero = inverse_softplus(y_near_zero)
    assert torch.isfinite(x_near_zero).all()


@pytest.mark.parametrize("threshold", [20.0, 5.0])
def test_inverse_softplus_above_threshold_is_identity(threshold):
    """Values above `threshold` bypass the log path and return unchanged."""
    y_high = torch.tensor([threshold + 5.0, threshold + 30.0])
    x_high = inverse_softplus(y_high, threshold=threshold)
    torch.testing.assert_close(y_high, x_high)
