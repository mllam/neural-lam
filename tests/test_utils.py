# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam.utils import inverse_sigmoid, inverse_softplus


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_inverse_sigmoid_roundtrip(dtype):
    """`inverse_sigmoid` recovers the input of `sigmoid`."""
    x_orig = torch.linspace(-5, 5, steps=100, dtype=dtype)
    y = torch.sigmoid(x_orig)
    x_reconstructed = inverse_sigmoid(y)

    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(x_orig, x_reconstructed, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_inverse_sigmoid_boundary(dtype):
    """Outputs and gradients stay finite near 0.0 and 1.0 across dtypes."""
    y_lower = torch.linspace(0.0, 1e-3, steps=10, dtype=dtype)
    y_upper = torch.linspace(1.0 - 1e-3, 1.0, steps=10, dtype=dtype)

    y_tensor = torch.cat([y_lower, y_upper]).requires_grad_(True)

    x = inverse_sigmoid(y_tensor)
    x.sum().backward()
    assert torch.isfinite(x).all()
    assert torch.isfinite(y_tensor.grad).all()
