# Standard library
import warnings
from unittest.mock import MagicMock

# Third-party
import pytest
import torch

# First-party
from neural_lam import lr_scheduler


@pytest.fixture
def model():
    return torch.nn.Linear(1, 1)


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)  # Real optimizer


def test_warmup_cosine_annealing_can_instantiate(optimizer):
    lrs = lr_scheduler.WarmupCosineAnnealingLR(optimizer, max_steps=1000)
    __import__("pdb").set_trace()  # TODO delme
