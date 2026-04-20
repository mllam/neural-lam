# Standard library
import os

# Third-party
import pytest
from pytorch_lightning.utilities import rank_zero_only

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(autouse=True)
def ensure_rank_zero(monkeypatch):
    """Ensure rank_zero_only.rank == 0 so @rank_zero_only-decorated functions
    execute their body regardless of state left by prior training tests."""
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
