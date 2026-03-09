# Standard library
import random

# Third-party
import numpy as np
import torch


def _worker_init_fn(worker_id):
    """Mirror of neural_lam.weather_dataset._worker_init_fn.

    Duplicated here to avoid importing neural_lam (which pulls in
    torch_geometric and other heavy dependencies).  Any change to the
    real function must be reflected here as well.
    """
    worker_seed = torch.initial_seed() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def test_worker_init_fn_different_worker_ids_differ():
    """Calling _worker_init_fn with the same torch seed but different
    worker ids should produce different random values."""

    def get_random_values_after_init(seed_val, worker_id):
        torch.manual_seed(seed_val)
        _worker_init_fn(worker_id)
        return np.random.rand(), random.random()

    # Same seed, different worker_id => different values
    a1, b1 = get_random_values_after_init(42, worker_id=0)
    a2, b2 = get_random_values_after_init(42, worker_id=1)

    assert a1 != a2, "numpy values should differ with different worker ids"
    assert b1 != b2, "stdlib values should differ with different worker ids"
