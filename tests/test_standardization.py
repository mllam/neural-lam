# Third-party
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# First-party
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example


def _get_datasets():
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)
    dataset_std = WeatherDataset(
        datastore=datastore, split="train", standardize=True
    )
    dataset_raw = WeatherDataset(
        datastore=datastore, split="train", standardize=False
    )
    return dataset_std, dataset_raw


def verify_standardization(dataset_std, dataset_raw, idx):
    # Fetch standardized and raw samples
    sample_std = dataset_std[idx]
    sample_raw = dataset_raw[idx]

    # Extract states
    init_std = sample_std[0]
    init_raw = sample_raw[0]

    target_std = sample_std[1]
    target_raw = sample_raw[1]

    # Fetch standardization stats and safely reshape them manually
    # stats are (d_features,). Broadcast with (N_times, N_grid, d_features)
    mean = torch.tensor(
        dataset_std.da_state_mean.values, dtype=torch.float32
    ).view(1, 1, -1)
    std = torch.tensor(
        dataset_std.da_state_std.values, dtype=torch.float32
    ).view(1, 1, -1)

    # Reconstruct
    reconstructed_init = init_std * std + mean
    reconstructed_target = target_std * std + mean

    # Assert correctness
    assert torch.allclose(
        reconstructed_init, init_raw, atol=1e-5
    ), "Init state standardization is not reversible"
    assert torch.allclose(
        reconstructed_target, target_raw, atol=1e-5
    ), "Target state standardization is not reversible"


def test_standardization_basic():
    dataset_std, dataset_raw = _get_datasets()
    verify_standardization(dataset_std, dataset_raw, idx=0)


@given(st.integers(min_value=0, max_value=2))
@settings(deadline=None, max_examples=3)
def test_standardization_property(idx):
    dataset_std, dataset_raw = _get_datasets()
    idx = min(idx, len(dataset_std) - 1)
    verify_standardization(dataset_std, dataset_raw, idx=idx)
