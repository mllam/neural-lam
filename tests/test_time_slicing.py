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


ANALYSIS_STATE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FORCING_VALUES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Constants for forecast data
FORECAST_ANALYSIS_TIMES = np.datetime64("2020-01-01") + np.arange(3)
FORECAST_FORECAST_TIMES = np.timedelta64(0, "D") + np.arange(7)

FORECAST_STATE_VALUES = np.array(
    [
        # Analysis time 0
        [0, 1, 2, 3, 4, 5, 6],
        # Analysis time 1
        [10, 11, 12, 13, 14, 15, 16],
        # Analysis time 2
        [20, 21, 22, 23, 24, 25, 26],
    ]
)

FORECAST_FORCING_VALUES = np.array(
    [
        # Analysis time 0
        [100, 101, 102, 103, 104, 105, 106],
        # Analysis time 1
        [110, 111, 112, 113, 114, 115, 116],
        # Analysis time 2
        [120, 121, 122, 123, 124, 125, 126],
    ]
)


@pytest.mark.parametrize(
    "ar_steps,num_past_forcing_steps,num_future_forcing_steps",
    [[3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0]],
)
def test_time_slicing_analysis(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    # state and forcing variables have only one dimension, `time`
    time_values = np.datetime64("2020-01-01") + np.arange(len(ANALYSIS_STATE_VALUES))
    assert len(ANALYSIS_STATE_VALUES) == len(FORCING_VALUES) == len(time_values)

    datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
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

    init_states, target_states, forcing, _, _ = [tensor.numpy() for tensor in sample]

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
    # forcing: (ar_steps, N_grid, d_windowed_forcing * 2)
    # target_times: (ar_steps,)
    assert init_states.shape == (2, 1, 1)
    assert init_states[:, 0, 0].tolist() == expected_init_states

    assert target_states.shape == (3, 1, 1)
    assert target_states[:, 0, 0].tolist() == expected_target_states

    assert forcing.shape == (
        3,
        1,
        # Factor 2 because each window step has a temporal embedding
        (1 + num_past_forcing_steps + num_future_forcing_steps) * 2,
    )
    np.testing.assert_equal(
        forcing[:, 0, : num_past_forcing_steps + num_future_forcing_steps + 1],
        np.array(expected_forcing_values),
    )


@pytest.mark.parametrize(
    "ar_steps,num_past_forcing_steps,num_future_forcing_steps",
    [
        [3, 0, 0],
        [3, 1, 0],
        [3, 2, 0],
        [3, 0, 1],
        [3, 0, 2],
    ],
)
def test_time_slicing_forecast(
    ar_steps, num_past_forcing_steps, num_future_forcing_steps
):
    # Create a dummy datastore with forecast data
    time_values = (FORECAST_ANALYSIS_TIMES, FORECAST_FORECAST_TIMES)
    datastore = SinglePointDummyDatastore(
        state_data=FORECAST_STATE_VALUES,
        forcing_data=FORECAST_FORCING_VALUES,
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
    assert len(dataset) == len(FORECAST_ANALYSIS_TIMES)

    sample = dataset[0]

    init_states, target_states, forcing, _, _ = [tensor.numpy() for tensor in sample]

    # Expected initial states and target states
    expected_init_states = FORECAST_STATE_VALUES[0][:2]
    expected_target_states = FORECAST_STATE_VALUES[0][2 : 2 + ar_steps]

    # Expected forcing values
    total_forcing_window = num_past_forcing_steps + num_future_forcing_steps + 1
    expected_forcing_values = []
    for i in range(ar_steps):
        start_idx = max(0, i + 2 - num_past_forcing_steps)
        end_idx = i + 2 + num_future_forcing_steps + 1
        forcing_window = FORECAST_FORCING_VALUES[0][start_idx:end_idx]
        expected_forcing_values.append(forcing_window)

    # Assertions
    np.testing.assert_array_equal(init_states[:, 0, 0], expected_init_states)
    np.testing.assert_array_equal(target_states[:, 0, 0], expected_target_states)

    # Verify the shape of the forcing data
    expected_forcing_shape = (
        ar_steps,
        1,
        total_forcing_window * 2,  # Each windowed feature includes temporal embedding
    )
    assert forcing.shape == expected_forcing_shape

    # Extract the forcing values from the tensor (excluding temporal embeddings)
    forcing_values = forcing[:, 0, :total_forcing_window]

    # Compare with expected forcing values
    for i in range(ar_steps):
        np.testing.assert_array_equal(forcing_values[i], expected_forcing_values[i])
