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
    """One-grid-point datastore in either analysis or forecast mode.

    Analysis mode: ``time_values`` is a 1D datetime array, ``state_data``
    and ``forcing_data`` are 1D arrays aligned to it.

    Forecast mode: ``time_values`` is the pair
    ``(analysis_times, elapsed_forecast_durations)``, and ``state_data``
    / ``forcing_data`` are 2D arrays shaped
    ``(n_analysis_times, n_forecast_steps)``.
    """

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
        self._state_data = np.array(state_data)
        self._forcing_data = np.array(forcing_data)
        self.is_forecast = is_forecast

        if is_forecast:
            self._analysis_times = np.array(time_values[0])
            self._forecast_times = np.array(time_values[1])
            assert self._state_data.ndim == 2
        else:
            self._time_values = np.array(time_values)
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
            da = xr.DataArray(
                values,
                dims=["analysis_time", "elapsed_forecast_duration"],
                coords={
                    "analysis_time": self._analysis_times,
                    "elapsed_forecast_duration": self._forecast_times,
                },
            )
        else:
            da = xr.DataArray(
                values, dims=["time"], coords={"time": self._time_values}
            )

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
# Boundary spans 4 extra steps on each side of the interior so windowing
# with up to num_past/num_future = 4 can be tested without cropping.
BOUNDARY_PAD = 4
BOUNDARY_FORCING_VALUES = list(range(20, 20 + 10 + 2 * BOUNDARY_PAD))

FORECAST_STATE_VALUES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
]
FORECAST_FORCING_VALUES = [
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    [120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
    [130, 131, 132, 133, 134, 135, 136, 137, 138, 139],
]


class BoundaryOnlyDummyDatastore(SinglePointDummyDatastore):
    """Boundary-only variant providing forcing but no state.

    State-keyed lookups raise KeyError to mirror real boundary datastores
    (e.g. ERA5) and to catch any path that accidentally asks the boundary
    for state.
    """

    def __init__(
        self,
        time_values,
        forcing_data,
        is_forecast=False,
        step_length=timedelta(hours=1),
    ):
        # state_data is a dummy zeros array of the right shape so the
        # parent constructor accepts it; the override below blocks state
        # access.
        forcing_arr = np.asarray(forcing_data)
        super().__init__(
            time_values=time_values,
            state_data=np.zeros_like(forcing_arr),
            forcing_data=forcing_arr,
            is_forecast=is_forecast,
            step_length=step_length,
        )

    def get_dataarray(self, category, split):
        if category == "state":
            raise KeyError("BoundaryOnlyDummyDatastore has no state category.")
        return super().get_dataarray(category=category, split=split)


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
    )

    sample = dataset[0]

    init_states, target_states, forcing, _boundary, _ = [
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
    )

    # Test that we can get a sample
    sample = dataset[0]
    assert (
        len(sample) == 5
    )  # init_states, target_states, forcing, boundary, target_times


def _interior_times():
    return np.datetime64("2020-01-01") + np.arange(len(ANALYSIS_STATE_VALUES))


def _boundary_times_aligned():
    """Boundary times surrounding the interior on both sides so that
    windows up to BOUNDARY_PAD steps don't trigger cropping."""
    return (
        np.datetime64("2020-01-01")
        - BOUNDARY_PAD
        + np.arange(len(BOUNDARY_FORCING_VALUES))
    )


@pytest.mark.parametrize(
    "ar_steps,num_past_boundary_steps,num_future_boundary_steps",
    [
        [3, 0, 0],
        [3, 1, 0],
        [3, 0, 1],
        [3, 1, 1],
        [3, 2, 2],
        [3, 3, 1],
        [3, 1, 3],
    ],
)
def test_time_slicing_boundary_analysis(
    ar_steps, num_past_boundary_steps, num_future_boundary_steps
):
    """Boundary windowing for analysis-interior + analysis-boundary.

    Boundary spans BOUNDARY_PAD extra steps on each side of the interior
    so no cropping kicks in; the exact window values around each state
    time are checked."""
    interior_datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=_interior_times(),
        is_forecast=False,
    )
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=BOUNDARY_FORCING_VALUES,
        time_values=_boundary_times_aligned(),
        is_forecast=False,
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=ar_steps,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=num_past_boundary_steps,
        num_future_boundary_steps=num_future_boundary_steps,
    )

    _, _, _, boundary, _ = [tensor.numpy() for tensor in dataset[0]]

    # Interior sample idx=0 has state slice [t_0..t_4] (no past-forcing
    # offset since num_past_forcing=0). Target states start at t_2; the
    # boundary index for t_2 in BOUNDARY_FORCING_VALUES is BOUNDARY_PAD+2.
    boundary_center = BOUNDARY_PAD + 2
    window_size = num_past_boundary_steps + num_future_boundary_steps + 1
    assert boundary.shape == (ar_steps, 1, window_size)
    for i in range(ar_steps):
        start = boundary_center + i - num_past_boundary_steps
        end = boundary_center + i + num_future_boundary_steps + 1
        expected = BOUNDARY_FORCING_VALUES[start:end]
        np.testing.assert_array_equal(boundary[i, 0, :], expected)


def test_boundary_step_length_mismatch_supported():
    """Interior and boundary with different step lengths align by time:
    a 6h boundary still produces correctly-windowed slices around the
    1h interior times."""
    interior_times = np.datetime64("2020-01-01") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    interior_values = np.arange(24, dtype=float)

    # Boundary every 6h, covering the same calendar span plus a 6h pad
    # on each end so the past/future window stays in-bounds.
    boundary_times = np.datetime64("2019-12-31T18:00") + np.arange(
        7
    ) * np.timedelta64(6, "h")
    boundary_values = np.arange(100, 107, dtype=float)

    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=1),
    )
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=boundary_times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    _, _, _, boundary, _ = [tensor.numpy() for tensor in dataset[0]]
    # First target state is at hour 2; nearest boundary <= hour 2 is hour 0
    # (= boundary_values[1] = 101). Window [past=1, future=1] takes
    # boundary_values[0], boundary_values[1], boundary_values[2].
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(boundary[0, 0, :], [100, 101, 102])
    np.testing.assert_array_equal(boundary[1, 0, :], [100, 101, 102])


def test_forecast_interior_with_analysis_boundary():
    """Forecast-mode interior + analysis-mode boundary: boundary windows
    around each lead-time of the forecast pick the corresponding boundary
    times."""
    analysis_times = np.datetime64("2020-01-01") + np.arange(
        len(FORECAST_STATE_VALUES)
    ) * np.timedelta64(1, "D")
    forecast_durations = np.arange(
        len(FORECAST_STATE_VALUES[0])
    ) * np.timedelta64(1, "D")

    interior_datastore = SinglePointDummyDatastore(
        state_data=FORECAST_STATE_VALUES,
        forcing_data=FORECAST_FORCING_VALUES,
        time_values=(analysis_times, forecast_durations),
        is_forecast=True,
        step_length=timedelta(days=1),
    )

    # Boundary covers analysis_time[0] + leads, padded on both sides.
    boundary_times = np.datetime64("2019-12-30") + np.arange(
        12
    ) * np.timedelta64(1, "D")
    boundary_values = np.arange(200, 212, dtype=float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=boundary_times,
        is_forecast=False,
        step_length=timedelta(days=1),
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=3,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    init_states, target_states, _, boundary, _ = [t.numpy() for t in dataset[0]]
    # Sample idx=0: pick analysis_time[0] (2020-01-01), state at lead
    # 0..4 = [0,1,2,3,4]. Init=[0,1], target=[2,3,4]. State times are
    # 2020-01-01 + (0..4) days = 01..05.
    np.testing.assert_array_equal(init_states[:, 0, 0], [0, 1])
    np.testing.assert_array_equal(target_states[:, 0, 0], [2, 3, 4])
    # Boundary starts at 2019-12-30 (idx 0). Target state times 03..05
    # correspond to boundary idx 4..6, with past/future windows of 1.
    assert boundary.shape == (3, 1, 3)
    np.testing.assert_array_equal(boundary[0, 0, :], [203, 204, 205])
    np.testing.assert_array_equal(boundary[1, 0, :], [204, 205, 206])
    np.testing.assert_array_equal(boundary[2, 0, :], [205, 206, 207])


def test_analysis_interior_with_forecast_boundary():
    """Analysis-mode interior + forecast-mode boundary: an analysis time
    of the boundary forecast is picked so the requested past/future
    window around each target state time stays in lead-range, then
    lead-time windows are walked across AR steps."""
    interior_times = np.datetime64("2020-01-05") + np.arange(
        8
    ) * np.timedelta64(1, "D")
    interior_values = np.arange(8, dtype=float)
    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(days=1),
    )

    # Boundary: 6 analysis times, 8 lead-day steps each. Analysis times
    # 2020-01-04..09 so coverage extends past the latest interior
    # target times after cropping.
    n_analysis = 6
    n_leads = 8
    boundary_analysis = np.datetime64("2020-01-04") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    boundary_leads = np.arange(n_leads) * np.timedelta64(1, "D")
    boundary_values = (
        np.arange(n_analysis).reshape(-1, 1) * 1000
        + np.arange(n_leads).reshape(1, -1) * 10
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(days=1),
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    _, _, _, boundary, _ = [t.numpy() for t in dataset[0]]
    # Sample idx=0: state slice = interior[0:4] = times 2020-01-05..08.
    # Targets are 2020-01-07 and 2020-01-08 (first_target=07). Boundary
    # analysis_time pad-pick for 07 = idx 3 (07); equals first_target so
    # decrement to idx 2 (06). lead_at_first_target = (07-06)/1d = 1,
    # which already covers num_past=1, so no further shift. Window at
    # target 07: lead 1, [0..2]. Window at target 08: lead 2, [1..3].
    expected_analysis_idx = 2
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(
        boundary[0, 0, :], boundary_values[expected_analysis_idx, 0:3]
    )
    np.testing.assert_array_equal(
        boundary[1, 0, :], boundary_values[expected_analysis_idx, 1:4]
    )


def test_check_time_overlap_insufficient_raises():
    """If the boundary cannot be cropped enough to cover the requested
    past-window, ``check_time_overlap`` surfaces a clear error."""
    interior_datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=_interior_times(),
        is_forecast=False,
    )
    # Boundary covers the same range as interior but no padding, so
    # any non-zero past/future window forces cropping; with a huge past
    # window the boundary cannot cover even a single sample.
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=BOUNDARY_FORCING_VALUES[:10],
        time_values=_interior_times(),
        is_forecast=False,
    )

    with pytest.raises(ValueError):
        WeatherDataset(
            datastore=interior_datastore,
            datastore_boundary=boundary_datastore,
            ar_steps=3,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=20,
            num_future_boundary_steps=20,
        )
