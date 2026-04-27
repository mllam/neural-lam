# Third-party
import torch

# First-party
from neural_lam.metrics import wmae, wmse


# ---------------------------------------------------------------------------
# Tests for softplus + epsilon fix (base_graph_model.py)
# ---------------------------------------------------------------------------


def test_softplus_epsilon_always_positive():
    """Softplus output + 1e-6 must be strictly positive for all inputs.

    For very negative inputs in float32, ``softplus(x)`` returns subnormal
    values close to 0. Dividing by a subnormal (or zero) propagates inf/NaN
    through metrics. The 1e-6 constant ensures the denominator is always at
    least 1e-6, well above the subnormal range.
    """
    # Inputs ranging from extremely negative to large positive
    raw = torch.tensor([-200.0, -100.0, -80.0, 0.0, 10.0])

    fixed_std = torch.nn.functional.softplus(raw) + 1e-6

    assert (fixed_std > 0.0).all(), (
        "pred_std with epsilon fix must be strictly positive for all inputs"
    )
    assert torch.isfinite(fixed_std).all(), (
        "pred_std with epsilon fix must be finite for all inputs"
    )
    # Must be comfortably above subnormal range
    assert (fixed_std >= 1e-7).all(), (
        "pred_std must stay well above the float32 subnormal range to avoid "
        "numerical instability when used as a divisor"
    )


def test_wmse_finite_with_softplus_epsilon():
    """wmse must return a finite value when pred_std uses the epsilon fix.

    This is an end-to-end check of the loss integration path: generate a
    pred_std from a very negative raw output, apply the fix, and confirm the
    resulting wmse value is finite and not NaN.
    """
    B, N, d = 2, 16, 3
    pred = torch.randn(B, N, d)
    target = torch.randn(B, N, d)
    raw_std = torch.full((B, N, d), -100.0)

    # With the epsilon fix
    pred_std = torch.nn.functional.softplus(raw_std) + 1e-6
    loss = wmse(pred, target, pred_std)

    assert torch.isfinite(loss).all().item(), (
        f"wmse produced a non-finite value ({loss.tolist()}) even with the "
        "softplus epsilon fix applied"
    )


def test_wmae_finite_with_softplus_epsilon():
    """wmae must return a finite value when pred_std uses the epsilon fix."""
    B, N, d = 2, 16, 3
    pred = torch.randn(B, N, d)
    target = torch.randn(B, N, d)
    raw_std = torch.full((B, N, d), -100.0)

    pred_std = torch.nn.functional.softplus(raw_std) + 1e-6
    loss = wmae(pred, target, pred_std)

    assert torch.isfinite(loss).all().item(), (
        f"wmae produced a non-finite value ({loss.tolist()}) even with the "
        "softplus epsilon fix applied"
    )


# ---------------------------------------------------------------------------
# Tests for per_var_std + zero feature-weight fix (ar_model.py)
# ---------------------------------------------------------------------------


def _make_feature_weights(n_vars, zero_indices=()):
    """Return a float32 tensor of uniform weights with specified indices set to 0."""
    w = torch.ones(n_vars, dtype=torch.float32)
    for i in zero_indices:
        w[i] = 0.0
    return w


def test_per_var_std_finite_with_zero_weight():
    """diff_std / sqrt(feature_weights + 1e-8) must be finite even for w=0.

    Without the epsilon, ``sqrt(0) == 0`` and the division yields inf.
    """
    n_vars = 5
    diff_std = torch.ones(n_vars, dtype=torch.float32)  # normalised

    # Set the last variable's weight to 0
    feature_weights = _make_feature_weights(n_vars, zero_indices=[-1])

    # Compute per_var_std using the fixed formula
    per_var_std = diff_std / torch.sqrt(feature_weights + 1e-8)

    assert torch.isfinite(per_var_std).all(), (
        "per_var_std contains inf or NaN when a feature weight is 0.0. "
        "The epsilon (1e-8) must prevent this."
    )


def test_per_var_std_inf_without_epsilon():
    """Confirm the unfixed formula produces inf when a weight is 0.

    This is a negative control test that documents the original bug: computing
    ``diff_std / sqrt(0)`` does produce infinity.
    """
    n_vars = 5
    diff_std = torch.ones(n_vars, dtype=torch.float32)
    feature_weights = _make_feature_weights(n_vars, zero_indices=[-1])

    # Without the fix
    per_var_std_broken = diff_std / torch.sqrt(feature_weights)

    assert not torch.isfinite(per_var_std_broken).all(), (
        "Expected the unfixed formula to produce inf for a zero weight, "
        "but it did not — test assumption may be wrong."
    )
