import torch
import torch.nn.functional as F
import pytest
from neural_lam.utils import inverse_softplus

def test_inverse_softplus():
    """Test the inverse_softplus numerical stability and accuracy."""
    
    # 1. Round-trip identity check (The most critical ML test)
    # Generate 100 random values between -10 and 10
    x_orig = torch.linspace(-10, 10, steps=100)
    
    # Apply softplus, then inverse
    y = F.softplus(x_orig)
    x_reconstructed = inverse_softplus(y)
    
    # Assert they are mathematically identical (within standard floating point error)
    torch.testing.assert_close(x_orig, x_reconstructed)

    # 2. Near-zero input (Testing numerical stability / avoiding NaNs)
    # If the clamping fails, log(0) will crash the model with a NaN
    y_near_zero = torch.tensor([1e-7, 1e-6])
    x_near_zero = inverse_softplus(y_near_zero)
    
    assert not torch.isnan(x_near_zero).any(), "Near-zero input produced NaN"
    assert not torch.isinf(x_near_zero).any(), "Near-zero input produced Inf"

    # 3. Above-threshold input (Testing the linear passthrough branch)
    # Default threshold is 20, so 25 and 50 should bypass the log math
    y_high = torch.tensor([25.0, 50.0]) 
    x_high = inverse_softplus(y_high)
    
    torch.testing.assert_close(y_high, x_high, msg="Above-threshold values did not pass through linearly")