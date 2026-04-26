# Standard library
import os

# Third-party
import pytest
from pytorch_lightning.utilities import rank_zero_only

# First-party
from neural_lam.datastore import DATASTORES

# Local
from .dummy_datastore import DummyDatastore

# Disable weights and biases to avoid unnecessary logging
# and to avoid having to deal with authentication
os.environ["WANDB_MODE"] = "disabled"

# Register DummyDatastore so DatastoreSelection validation passes
# in any test that uses it (unit or integration)
DATASTORES.setdefault(DummyDatastore.SHORT_NAME, DummyDatastore)


@pytest.fixture(autouse=True)
def ensure_rank_zero(monkeypatch):
    """Ensure rank_zero_only.rank == 0 so @rank_zero_only-decorated functions
    execute their body regardless of state left by prior training tests."""
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
