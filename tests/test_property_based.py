# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# First-party
from neural_lam.weather_dataset import WeatherDataset
from tests.test_time_slicing import SinglePointDummyDatastore

ANALYSIS_STATE_VALUES = list(range(30))  # bigger pool than original
FORCING_VALUES = list(range(100, 130))


def make_datastore(num_steps=30):
    time_values = np.datetime64("2020-01-01") + np.arange(num_steps)
    return SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES[:num_steps],
        forcing_data=FORCING_VALUES[:num_steps],
        time_values=time_values,
        is_forecast=False,
    )


# ── Time slicing invariants ───────────────────────────────────────────────────

@given(
    ar_steps=st.integers(min_value=1, max_value=5),
    num_past_forcing_steps=st.integers(min_value=0, max_value=4),
    num_future_forcing_steps=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=200)
def test_time_slicing_output_shapes_invariant(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    """
    Property: for any valid combination of ar_steps / forcing window,
    the dataset sample shapes are always consistent with the parameters.
    """
    datastore = make_datastore(num_steps=30)

    # minimum required timesteps for at least one valid sample
    min_required = 2 + ar_steps + num_past_forcing_steps + num_future_forcing_steps
    assume(min_required <= 30)

    dataset = WeatherDataset(
        datastore=datastore,
        ar_steps=ar_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        standardize=False,
    )

    assume(len(dataset) > 0)

    init_states, target_states, forcing, target_times = [
        t.numpy() for t in dataset[0]
    ]

    # init states are always 2 timesteps
    assert init_states.shape[0] == 2

    # target states length always equals ar_steps
    assert target_states.shape[0] == ar_steps

    # forcing window width = past + current + future
    expected_forcing_width = 1 + num_past_forcing_steps + num_future_forcing_steps
    assert forcing.shape[0] == ar_steps
    assert forcing.shape[2] == expected_forcing_width

    # target_times length always equals ar_steps
    assert target_times.shape[0] == ar_steps


@given(
    ar_steps=st.integers(min_value=1, max_value=5),
    num_past_forcing_steps=st.integers(min_value=0, max_value=4),
    num_future_forcing_steps=st.integers(min_value=0, max_value=4),
    sample_idx=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=200)
def test_time_slicing_sample_index_invariant(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps, sample_idx
):
    """
    Property: any valid sample index always returns correctly shaped output,
    not just index 0.
    """
    datastore = make_datastore(num_steps=30)
    min_required = 2 + ar_steps + num_past_forcing_steps + num_future_forcing_steps
    assume(min_required <= 30)

    dataset = WeatherDataset(
        datastore=datastore,
        ar_steps=ar_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        standardize=False,
    )

    assume(sample_idx < len(dataset))

    init_states, target_states, forcing, target_times = [
        t.numpy() for t in dataset[sample_idx]
    ]

    assert init_states.shape[0] == 2
    assert target_states.shape[0] == ar_steps
    assert forcing.shape[0] == ar_steps
    assert target_times.shape[0] == ar_steps


@given(
    step_length_hours=st.integers(min_value=1, max_value=24),
)
def test_step_length_invariant(step_length_hours):
    """
    Property: any positive step_length always produces a usable dataset.
    """
    step_length = timedelta(hours=step_length_hours)
    time_values = np.datetime64("2020-01-01") + np.arange(20)
    datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES[:20],
        forcing_data=FORCING_VALUES[:20],
        time_values=time_values,
        is_forecast=False,
        step_length=step_length,
    )

    assert datastore.step_length == step_length

    dataset = WeatherDataset(
        datastore=datastore,
        ar_steps=3,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        standardize=False,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert len(sample) == 4