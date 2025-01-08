# Third-party
import numpy as np
import pytest
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore
from neural_lam.weather_dataset import WeatherDataset


class SinglePointDummyDatastore(BaseDatastore):
    step_length = 1
    config = None
    coords_projection = None
    num_grid_points = 1
    root_path = None

    def __init__(self, time_values, state_data, forcing_data, is_forecast):
        self.is_forecast = is_forecast
        if is_forecast:
            self._analysis_times, self._forecast_times = time_values
            self._state_data = np.array(state_data)
            self._forcing_data = np.array(forcing_data)
            # state_data and forcing_data should be 2D arrays with shape
            # (n_analysis_times, n_forecast_times)
        else:
            self._time_values = np.array(time_values)
            self._state_data = np.array(state_data)
            self._forcing_data = np.array(forcing_data)

            if is_forecast:
                assert self._state_data.ndim == 2
            else:
                assert self._state_data.ndim == 1

    def get_num_data_vars(self, category):
        return 1

    def get_dataarray(self, category, split):
        if self.is_forecast:
            if category == "state":
                # Create DataArray with dims ('analysis_time',
                # 'elapsed_forecast_duration')
                da = xr.DataArray(
                    self._state_data,
                    dims=["analysis_time", "elapsed_forecast_duration"],
                    coords={
                        "analysis_time": self._analysis_times,
                        "elapsed_forecast_duration": self._forecast_times,
                    },
                )
            elif category == "forcing":
                da = xr.DataArray(
                    self._forcing_data,
                    dims=["analysis_time", "elapsed_forecast_duration"],
                    coords={
                        "analysis_time": self._analysis_times,
                        "elapsed_forecast_duration": self._forecast_times,
                    },
                )
            else:
                raise NotImplementedError(category)
            # Add 'grid_index' and '{category}_feature' dimensions
            da = da.expand_dims("grid_index")
            da = da.expand_dims(f"{category}_feature")
            dim_order = self.expected_dim_order(category=category)
            return da.transpose(*dim_order)
        else:
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


INIT_STEPS = 2

STATE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FORCING_VALUES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

STATE_VALUES_FORECAST = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Analysis time 0
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # Analysis time 1
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  # Analysis time 2
]
FORCING_VALUES_FORECAST = [
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],  # Analysis time 0
    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],  # Analysis time 1
    [120, 121, 122, 123, 124, 125, 126, 127, 128, 129],  # Analysis time 2
]

SCENARIOS = [
    [3, 0, 0],
    [3, 1, 0],
    [3, 2, 0],
    [3, 3, 0],
    [3, 0, 1],
    [3, 0, 2],
    [3, 0, 3],
    [3, 1, 1],
    [3, 2, 1],
    [3, 3, 1],
    [3, 1, 2],
    [3, 1, 3],
    [3, 2, 2],
    [3, 2, 3],
    [3, 3, 2],
    [3, 3, 3],
]


@pytest.mark.parametrize(
    "ar_steps,num_past_forcing_steps,num_future_forcing_steps",
    SCENARIOS,
)
def test_time_slicing_analysis(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    # state and forcing variables have only one dimension, `time`
    time_values = np.datetime64("2020-01-01") + np.arange(len(STATE_VALUES))
    assert len(STATE_VALUES) == len(FORCING_VALUES) == len(time_values)

    datastore = SinglePointDummyDatastore(
        state_data=STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=time_values,
        is_forecast=False,
    )

    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        ar_steps=ar_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        standardize=False,
    )

    sample = dataset[0]

    init_states, target_states, forcing, _, _ = [
        tensor.numpy() for tensor in sample
    ]

    # Some scenarios for the human reader
    expected_init_states = [0, 1]
    if ar_steps == 3:
        expected_target_states = [2, 3, 4]
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

    # Compute expected initial states and target states based on ar_steps
    offset = max(0, num_past_forcing_steps - INIT_STEPS)
    init_idx = INIT_STEPS + offset
    # Compute expected forcing values based on num_past_forcing_steps and
    # num_future_forcing_steps for all scenarios
    expected_init_states = STATE_VALUES[offset:init_idx]
    expected_target_states = STATE_VALUES[init_idx : init_idx + ar_steps]
    total_forcing_window = num_past_forcing_steps + num_future_forcing_steps + 1
    expected_forcing_values = []
    for i in range(ar_steps):
        start_idx = i + init_idx - num_past_forcing_steps
        end_idx = i + init_idx + num_future_forcing_steps + 1
        forcing_window = FORCING_VALUES[start_idx:end_idx]
        expected_forcing_values.append(forcing_window)

    # init_states: (2, N_grid, d_features)
    # target_states: (ar_steps, N_grid, d_features)
    # forcing: (ar_steps, N_grid, d_windowed_forcing * 2)
    # target_times: (ar_steps,)

    # Adjust assertions to use computed expected values
    assert init_states.shape == (INIT_STEPS, 1, 1)
    np.testing.assert_array_equal(init_states[:, 0, 0], expected_init_states)

    assert target_states.shape == (ar_steps, 1, 1)
    np.testing.assert_array_equal(
        target_states[:, 0, 0], expected_target_states
    )

    assert forcing.shape == (
        ar_steps,
        1,
        total_forcing_window * 2,  # Each windowed feature includes time deltas
    )

    # Extract the forcing values from the tensor (excluding time deltas)
    forcing_values = forcing[:, 0, :total_forcing_window]

    # Compare with expected forcing values
    for i in range(ar_steps):
        np.testing.assert_array_equal(
            forcing_values[i], expected_forcing_values[i]
        )


@pytest.mark.parametrize(
    "ar_steps,num_past_forcing_steps,num_future_forcing_steps",
    SCENARIOS,
)
def test_time_slicing_forecast(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    # Constants for forecast data
    ANALYSIS_TIMES = np.datetime64("2020-01-01") + np.arange(
        len(STATE_VALUES_FORECAST)
    )
    ELAPSED_FORECAST_DURATION = np.timedelta64(0, "D") + np.arange(
        # Retrieving the first analysis_time
        len(FORCING_VALUES_FORECAST[0])
    )
    # Create a dummy datastore with forecast data
    time_values = (ANALYSIS_TIMES, ELAPSED_FORECAST_DURATION)
    datastore = SinglePointDummyDatastore(
        state_data=STATE_VALUES_FORECAST,
        forcing_data=FORCING_VALUES_FORECAST,
        time_values=time_values,
        is_forecast=True,
    )

    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=None,
        split="train",
        ar_steps=ar_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        standardize=False,
    )

    # Test the dataset length
    assert len(dataset) == len(ANALYSIS_TIMES)

    sample = dataset[0]

    init_states, target_states, forcing, _, _ = [
        tensor.numpy() for tensor in sample
    ]

    # Compute expected initial states and target states based on ar_steps
    offset = max(0, num_past_forcing_steps - INIT_STEPS)
    init_idx = INIT_STEPS + offset
    # Retrieving the first analysis_time
    expected_init_states = STATE_VALUES_FORECAST[0][offset:init_idx]
    expected_target_states = STATE_VALUES_FORECAST[0][
        init_idx : init_idx + ar_steps
    ]

    # Compute expected forcing values based on num_past_forcing_steps and
    # num_future_forcing_steps
    total_forcing_window = num_past_forcing_steps + num_future_forcing_steps + 1
    expected_forcing_values = []
    for i in range(ar_steps):
        start_idx = i + init_idx - num_past_forcing_steps
        end_idx = i + init_idx + num_future_forcing_steps + 1
        # Retrieving the analysis_time relevant for forcing-windows (i.e.
        # the first analysis_time after the 2 init_steps)
        forcing_window = FORCING_VALUES_FORECAST[INIT_STEPS][start_idx:end_idx]
        expected_forcing_values.append(forcing_window)

    # init_states: (2, N_grid, d_features)
    # target_states: (ar_steps, N_grid, d_features)
    # forcing: (ar_steps, N_grid, d_windowed_forcing * 2)
    # target_times: (ar_steps,)

    # Assertions
    np.testing.assert_array_equal(init_states[:, 0, 0], expected_init_states)
    np.testing.assert_array_equal(
        target_states[:, 0, 0], expected_target_states
    )

    # Verify the shape of the forcing data
    expected_forcing_shape = (
        ar_steps,  # Number of AR steps
        1,  # Number of grid points
        total_forcing_window  # Total number of forcing steps in the window
        * 2,  # Each windowed feature includes time deltas
    )
    assert forcing.shape == expected_forcing_shape

    # Extract the forcing values from the tensor (excluding time deltas)
    forcing_values = forcing[:, 0, :total_forcing_window]

    # Compare with expected forcing values
    for i in range(ar_steps):
        np.testing.assert_array_equal(
            forcing_values[i], expected_forcing_values[i]
        )
