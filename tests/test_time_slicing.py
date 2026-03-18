# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
import numpy as np
import pytest
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore
from neural_lam.weather_dataset import WeatherDataset


class SinglePointDummyDatastore(BaseDatastore):
    config = {}
    coords_projection = None
    num_grid_points = 1
    root_path = Path("dummy")

    def __init__(
        self,
        time_values,
        state_data,
        forcing_data,
        is_forecast,
        step_length=timedelta(hours=1),
    ):
        self._step_length = step_length
        self._time_values = np.array(time_values)
        self._state_data = np.array(state_data)
        self._forcing_data = np.array(forcing_data)
        self.is_forecast = is_forecast

        if is_forecast:
            assert self._state_data.ndim == 2
        else:
            assert self._state_data.ndim == 1

    @property
    def step_length(self):
        return self._step_length

    def get_num_data_vars(self, category):
        return 1

    def get_dataarray(self, category, split):
        if category == "state":
            values = self._state_data
        elif category == "forcing":
            values = self._forcing_data
        else:
            raise NotImplementedError(category)

        if self.is_forecast:
            raise NotImplementedError()
        else:
            da = xr.DataArray(
                values, dims=["time"], coords={"time": self._time_values}
            )
            # add `{category}_feature` and `grid_index` dimensions

        da = da.expand_dims("grid_index")
        da = da.expand_dims(f"{category}_feature")

        dim_order = self.expected_dim_order(category=category)
        return da.transpose(*dim_order)

    def get_standardization_dataarray(self, category):
        raise NotImplementedError()

    def get_xy(self, category):
        raise NotImplementedError()

    def get_vars_units(self, category):
        raise NotImplementedError()

    def get_vars_names(self, category):
        raise NotImplementedError()

    def get_vars_long_names(self, category):
        raise NotImplementedError()


ANALYSIS_STATE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FORCING_VALUES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


@pytest.mark.parametrize(
    "ar_steps,num_past_forcing_steps,num_future_forcing_steps",
    [[3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0]],
)
def test_time_slicing_analysis(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    # state and forcing variables have only on dimension, `time`
    time_values = np.datetime64("2020-01-01") + np.arange(
        len(ANALYSIS_STATE_VALUES)
    )
    assert len(ANALYSIS_STATE_VALUES) == len(FORCING_VALUES) == len(time_values)

    datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=time_values,
        is_forecast=False,
    )

    dataset = WeatherDataset(
        datastore=datastore,
        ar_steps=ar_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        standardize=False,
    )

    sample = dataset[0]

    init_states, target_states, forcing, _ = [
        tensor.numpy() for tensor in sample
    ]

    expected_init_states = [0, 1]
    if ar_steps == 3:
        expected_target_states = [2, 3, 4]
    else:
        raise NotImplementedError()

    if num_past_forcing_steps == num_future_forcing_steps == 0:
        expected_forcing_values = [[12], [13], [14]]
    elif num_past_forcing_steps == 1 and num_future_forcing_steps == 0:
        expected_forcing_values = [[11, 12], [12, 13], [13, 14]]
    elif num_past_forcing_steps == 2 and num_future_forcing_steps == 0:
        expected_forcing_values = [[10, 11, 12], [11, 12, 13], [12, 13, 14]]
    elif num_past_forcing_steps == 3 and num_future_forcing_steps == 0:
        expected_init_states = [1, 2]
        expected_target_states = [3, 4, 5]
        expected_forcing_values = [
            [10, 11, 12, 13],
            [11, 12, 13, 14],
            [12, 13, 14, 15],
        ]
    else:
        raise NotImplementedError()

    # init_states: (2, N_grid, d_features)
    # target_states: (ar_steps, N_grid, d_features)
    # forcing: (ar_steps, N_grid, d_windowed_forcing)
    # target_times: (ar_steps,)
    assert init_states.shape == (2, 1, 1)
    assert init_states[:, 0, 0].tolist() == expected_init_states

    assert target_states.shape == (3, 1, 1)
    assert target_states[:, 0, 0].tolist() == expected_target_states

    assert forcing.shape == (
        3,
        1,
        1 + num_past_forcing_steps + num_future_forcing_steps,
    )
    np.testing.assert_equal(forcing[:, 0, :], np.array(expected_forcing_values))


@pytest.mark.parametrize(
    "step_length",
    [timedelta(hours=1), timedelta(hours=3), timedelta(minutes=30)],
)
def test_step_length_timedeltas(step_length):
    """Test that datastores work with different step_length timedeltas."""
    time_values = np.datetime64("2020-01-01") + np.arange(
        len(ANALYSIS_STATE_VALUES)
    )
    datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=time_values,
        is_forecast=False,
        step_length=step_length,
    )

    # Test that the step_length property returns the correct timedelta
    assert datastore.step_length == step_length

    # Test that WeatherDataset can be created with this datastore
    dataset = WeatherDataset(
        datastore=datastore,
        ar_steps=3,
        num_future_forcing_steps=0,
        num_past_forcing_steps=0,
        standardize=False,
    )

    # Test that we can get a sample
    sample = dataset[0]
    assert len(sample) == 4  # init_states, target_states, forcing, target_times
