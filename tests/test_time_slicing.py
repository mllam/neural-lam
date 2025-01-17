# Third-party
import numpy as np
import pytest
import xarray as xr

# First-party
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseDatastore
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import (
    DATASTORES_BOUNDARY_EXAMPLES,
    init_datastore_boundary_example,
    init_datastore_example,
)


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


class BoundaryDummyDatastore(SinglePointDummyDatastore):
    """Dummy datastore with 6h timesteps for testing boundary conditions"""

    step_length = 6  # 6 hour timesteps


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
        total_forcing_window,  # No time deltas for interior forcing
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
        total_forcing_window,  # Total number of forcing steps in the window
        # no time deltas for interior forcing
    )
    assert forcing.shape == expected_forcing_shape

    # Extract the forcing values from the tensor (excluding time deltas)
    forcing_values = forcing[:, 0, :total_forcing_window]

    # Compare with expected forcing values
    for i in range(ar_steps):
        np.testing.assert_array_equal(
            forcing_values[i], expected_forcing_values[i]
        )


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize(
    "datastore_boundary_name", DATASTORES_BOUNDARY_EXAMPLES.keys()
)
@pytest.mark.parametrize(
    "subsample_config",
    [
        # (interior_subsample, boundary_subsample, ar_steps)
        (1, 1, 1),  # Base case - no subsampling
        (2, 1, 1),  # Interior subsampling only
        (1, 2, 1),  # Boundary subsampling only
        (2, 2, 1),  # Equal subsampling
        (2, 2, 2),  # More AR steps
    ],
)
def test_dataset_subsampling(
    datastore_name, datastore_boundary_name, subsample_config
):
    """Test that WeatherDataset handles different subsample steps correctly for
    interior and boundary data.

    The test checks:
    1. Dataset creation succeeds with different subsample configurations
    2. Time differences between consecutive states match subsample steps
    3. Shapes of returned tensors are correct
    4. We can access the last item without errors
    """
    interior_subsample, boundary_subsample, ar_steps = subsample_config

    datastore = init_datastore_example(datastore_name)
    datastore_boundary = init_datastore_boundary_example(
        datastore_boundary_name
    )

    # Configure dataset with subsampling
    dataset = WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        split="train",
        ar_steps=ar_steps,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        interior_subsample_step=interior_subsample,
        boundary_subsample_step=boundary_subsample,
    )

    # Get first sample
    init_states, target_states, forcing, boundary, target_times = dataset[0]

    # Check shapes
    assert init_states.shape[0] == 2  # Always 2 initial states
    assert target_states.shape[0] == ar_steps

    # Check time differences
    times = target_times.numpy()
    for i in range(1, len(times)):
        time_delta = np.timedelta64(times[i] - times[i - 1], "ns")
        expected_hours = interior_subsample * datastore.step_length
        np.testing.assert_equal(
            time_delta.astype("timedelta64[h]").astype(int), expected_hours
        )

    # Verify boundary data timesteps if present
    if boundary is not None:
        assert boundary.shape[0] == ar_steps
        # Each boundary window should have:
        # (num_past + num_future + 1) timesteps * features * 2 (for time deltas)
        expected_boundary_features = (
            datastore_boundary.get_num_data_vars("forcing") + 1
        ) * (
            1 + 1 + 1
        )  # past + future + current
        assert boundary.shape[2] == expected_boundary_features

    # Verify we can access the last item
    dataset[len(dataset) - 1]


@pytest.mark.parametrize(
    "num_past_steps,num_future_steps,interior_step,boundary_step",
    [
        (1, 1, 1, 1),  # Base case, no subsampling
        (2, 1, 1, 1),  # More past steps, no subsampling
        (1, 2, 1, 1),  # More future steps, no subsampling
        (2, 2, 1, 1),  # Equal past/future, no subsampling
        (1, 1, 1, 2),  # Basic case with boundary subsampling
        (2, 2, 1, 2),  # Equal past/future with boundary subsampling
        (1, 1, 2, 1),  # Basic case with interior subsampling
        (2, 2, 2, 1),  # Equal past/future with interior subsampling
        (1, 1, 2, 2),  # Both subsamplings
    ],
)
def test_time_deltas_in_boundary_data(
    num_past_steps, num_future_steps, interior_step, boundary_step
):
    """Test that time deltas are correctly calculated for boundary data.

    This test verifies:
    1. Time deltas are included in boundary data
    2. Time deltas are in units of state timesteps
    3. Time deltas are correctly calculated relative to current timestep
    4. Time steps scale correctly with subsampling
    """
    # Create dummy data with known timesteps (3 hour intervals for interior)
    time_values_interior = np.datetime64("2020-01-01") + np.arange(
        20
    ) * np.timedelta64(3, "h")
    # 6 hour intervals for boundary
    time_values_boundary = np.datetime64("2020-01-01") + np.arange(
        10
    ) * np.timedelta64(6, "h")

    time_step_ratio = (
        6 / 3
    )  # Boundary step is 6 hours, interior step is 3 hours

    state_data = np.arange(20)
    forcing_data = np.arange(20, 40)
    boundary_data = np.arange(10)  # Fewer points due to larger time step

    interior_datastore = SinglePointDummyDatastore(
        state_data=state_data,
        forcing_data=forcing_data,
        time_values=time_values_interior,
        is_forecast=False,
    )

    boundary_datastore = BoundaryDummyDatastore(
        state_data=boundary_data,
        forcing_data=boundary_data + 10,
        time_values=time_values_boundary,
        is_forecast=False,
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        split="train",
        ar_steps=2,
        num_past_boundary_steps=num_past_steps,
        num_future_boundary_steps=num_future_steps,
        interior_subsample_step=interior_step,
        boundary_subsample_step=boundary_step,
        standardize=False,
    )

    # Get first sample
    _, _, _, boundary, target_times = dataset[0]

    # Extract time deltas from boundary data
    # Time deltas are the last features in the boundary tensor
    window_size = num_past_steps + num_future_steps + 1
    time_deltas = boundary[0, 0, -window_size:].numpy()

    # Expected time deltas in state timesteps, adjusted for boundary subsampling
    # For each window position, calculate expected offset from current time
    expected_deltas = (
        np.arange(-num_past_steps, num_future_steps + 1)
        * boundary_step
        * time_step_ratio
    )

    # Verify time deltas match expected values
    np.testing.assert_array_equal(time_deltas, expected_deltas)

    # Calculate expected hours offset from current time
    # Each state timestep is 3 hours, scale by boundary step
    expected_hours = expected_deltas * boundary_datastore.step_length
    time_delta_hours = time_deltas * boundary_datastore.step_length

    # Verify time delta hours match expected values
    np.testing.assert_array_equal(time_delta_hours, expected_hours)

    # Verify relative hour differences between timesteps
    expected_hour_diff = (
        boundary_step * boundary_datastore.step_length * time_step_ratio
    )
    hour_diffs = np.diff(time_delta_hours)
    np.testing.assert_array_equal(
        hour_diffs, [expected_hour_diff] * (len(time_delta_hours) - 1)
    )

    # Extract boundary times and verify they match expected hours
    for i in range(len(target_times)):
        window_start_idx = i * (window_size * 2)
        window_end_idx = window_start_idx + window_size
        boundary_times = boundary[i, 0, window_start_idx:window_end_idx].numpy()
        boundary_time_diffs = (
            np.diff(boundary_times) * boundary_datastore.step_length
        )
        expected_diff = boundary_step * boundary_datastore.step_length
        np.testing.assert_array_equal(
            boundary_time_diffs, [expected_diff] * (len(boundary_times) - 1)
        )
