# Third-party
import torch

# Local
from neural_lam import metrics

def test_mask_and_reduce_without_grid_weights_matches_mean():
    # When no grid_weights are provided, behavior should match simple mean
    # This ensures backward compatibility with existing regional model usage
    vals = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.tensor([True, True, True])

    out = metrics.mask_and_reduce_metric(
        vals,
        mask=mask,
        average_grid=True,
        sum_vars=True,
        grid_weights=None,
    )

    # Simple mean of [1, 2, 3] = 2.0
    expected = torch.tensor([2.0])
    assert torch.isclose(out, expected)


def test_mask_and_reduce_with_grid_weights():
    # Weighted average: (1*1 + 2*1 + 3*2) / (1+1+2) = 9/4 = 2.25
    # Higher weight on node 3 pulls the average toward 3.0
    vals = torch.tensor([[[1.0], [2.0], [3.0]]])
    weights = torch.tensor([1.0, 1.0, 2.0])
    mask = torch.tensor([True, True, True])

    out = metrics.mask_and_reduce_metric(
        vals,
        mask=mask,
        average_grid=True,
        sum_vars=True,
        grid_weights=weights,
    )

    # Weighted mean: (1*1 + 2*1 + 3*2) / 4 = 2.25
    expected = torch.tensor([2.25])
    assert torch.isclose(out, expected)

def test_mask_and_reduce_with_mask_and_grid_weights():
    # Verifies that masking and weighting interact correctly
    # Node 3 has value 100 and weight 100 but is masked out
    # Only nodes 1 and 2 contribute, with equal weights -> mean = 1.5
    vals = torch.tensor([[[1.0], [2.0], [100.0]]])
    weights = torch.tensor([1.0, 1.0, 100.0])
    mask = torch.tensor([True, True, False])

    out = metrics.mask_and_reduce_metric(
        vals,
        mask=mask,
        average_grid=True,
        sum_vars=True,
        grid_weights=weights,
    )

    # Masked weighted mean: (1*1 + 2*1) / (1+1) = 1.5
    expected = torch.tensor([1.5])
    assert torch.isclose(out, expected)
