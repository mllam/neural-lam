# Third-party
import torch

# First-party
from neural_lam.metrics import crps_ensemble

def test_crps_shapes():
    preds = torch.randn(3, 2, 5, 10, 4)
    target = torch.randn(2, 5, 10, 4)

    out = crps_ensemble(preds, target)

    assert out.shape == target.shape
