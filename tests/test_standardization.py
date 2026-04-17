# Third-party
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# First-party
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example


def _get_dataset():
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)
    return WeatherDataset(
        datastore=datastore,
        split="train",
    )


def _collect_state_data(dataset, n_samples):
    """
    Collect standardized state tensors from WeatherDataset.
    """
    all_states = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        init_states = sample[0]
        target_states = sample[1]
        combined = torch.cat([init_states, target_states], dim=0)
        all_states.append(combined.detach().cpu().numpy())
    return np.concatenate(all_states, axis=0)


def test_standardization_basic():
    dataset = _get_dataset()
    data = _collect_state_data(dataset, n_samples=10)

    mean_per_feat = np.mean(data, axis=(0, 1))
    std_per_feat = np.std(data, axis=(0, 1))

    assert np.allclose(
        mean_per_feat, 0, atol=1e-1
    ), f"Expected means ~0, got {np.round(mean_per_feat, 4)}"
    assert np.allclose(
        std_per_feat, 1, atol=1e-1
    ), f"Expected stds ~1, got {np.round(std_per_feat, 4)}"


@given(st.integers(min_value=5, max_value=15))
@settings(deadline=None)
def test_standardization_property(n_samples):
    dataset = _get_dataset()
    data = _collect_state_data(dataset, n_samples=n_samples)

    mean_per_feat = np.mean(data, axis=(0, 1))
    std_per_feat = np.std(data, axis=(0, 1))

    assert np.allclose(
        mean_per_feat, 0, atol=1e-1
    ), f"n={n_samples}: expected means ~0, got {np.round(mean_per_feat, 4)}"
    assert np.allclose(
        std_per_feat, 1, atol=1e-1
    ), f"n={n_samples}: expected stds ~1, got {np.round(std_per_feat, 4)}"
