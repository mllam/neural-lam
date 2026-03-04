# Standard library
import os
import random

# Third-party
import numpy as np
import torch

# First-party
from neural_lam.weather_dataset import _worker_init_fn


def test_cublas_workspace_config_is_set():
    """Importing train_model should set CUBLAS_WORKSPACE_CONFIG."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    import importlib

    import neural_lam.train_model as tm

    importlib.reload(tm)

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_cublas_does_not_override_user_value():
    """If the user already set CUBLAS_WORKSPACE_CONFIG, we should not
    override it."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import importlib

    import neural_lam.train_model as tm

    importlib.reload(tm)

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"

    # Cleanup
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_worker_init_fn_seeds_deterministically():
    """Calling _worker_init_fn with the same torch seed should produce
    identical numpy and stdlib random values."""

    def get_random_values_after_init(seed_val, worker_id):
        torch.manual_seed(seed_val)
        _worker_init_fn(worker_id)
        return np.random.rand(), random.random()

    # Same seed, same worker_id => same values
    a1, b1 = get_random_values_after_init(42, worker_id=0)
    a2, b2 = get_random_values_after_init(42, worker_id=0)

    assert a1 == a2, "numpy values should match with same seed"
    assert b1 == b2, "stdlib values should match with same seed"


def test_worker_init_fn_different_seeds_differ():
    """Calling _worker_init_fn with different torch seeds should produce
    different random values."""

    def get_random_values_after_init(seed_val, worker_id):
        torch.manual_seed(seed_val)
        _worker_init_fn(worker_id)
        return np.random.rand(), random.random()

    a1, b1 = get_random_values_after_init(42, worker_id=0)
    a2, b2 = get_random_values_after_init(99, worker_id=0)

    assert a1 != a2, "numpy values should differ with different seeds"
    assert b1 != b2, "stdlib values should differ with different seeds"
