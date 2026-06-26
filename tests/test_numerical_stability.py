# Standard library
import warnings

# Third-party
import torch


def _per_var_std(feature_weights: torch.Tensor) -> torch.Tensor:
    """Mirror of the ForecasterModule per_var_std formula."""
    diff_std = torch.ones_like(feature_weights)
    eps = torch.finfo(torch.float32).eps
    return diff_std / torch.sqrt(feature_weights + eps)


def test_per_var_std_finite_with_zero_weight():
    """`per_var_std` must stay finite when a feature weight is exactly 0.

    Without the eps inside the sqrt, `diff_std / sqrt(0)` becomes `inf`,
    which propagates `NaN` through the weighted-MSE / weighted-MAE losses.
    """
    feature_weights = torch.tensor([1.0, 0.5, 0.0, 0.25], dtype=torch.float32)
    per_var_std = _per_var_std(feature_weights)

    assert torch.isfinite(per_var_std).all(), (
        f"per_var_std contained non-finite values for zero weight: "
        f"{per_var_std.tolist()}"
    )


def test_per_var_std_unchanged_for_nonzero_weights():
    """eps must not perturb non-zero feature weights at float32 precision."""
    feature_weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    diff_std = torch.ones_like(feature_weights)
    eps = torch.finfo(torch.float32).eps

    per_var_std_with_eps = diff_std / torch.sqrt(feature_weights + eps)
    per_var_std_no_eps = diff_std / torch.sqrt(feature_weights)

    torch.testing.assert_close(
        per_var_std_with_eps, per_var_std_no_eps, rtol=1e-6, atol=1e-6
    )


def test_zero_feature_weight_emits_warning():
    """A zero feature weight must trigger a `UserWarning` naming the index."""
    feature_weights = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32)
    zero_mask = feature_weights == 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if zero_mask.any().item():
            zero_indices = zero_mask.nonzero(as_tuple=False).squeeze(-1)
            warnings.warn(
                f"Feature weight(s) at indices {zero_indices.tolist()} "
                "are set to 0.0.",
                UserWarning,
                stacklevel=2,
            )

    assert len(caught) == 1
    assert "indices [1]" in str(caught[0].message)
