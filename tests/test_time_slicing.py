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

    def get_dataarray(self, category, split):  # type: ignore[override]
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

    def get_xy(self, category):  # type: ignore[override]
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
    # Model init is the last input state 2020-01-06; targets are 07 and 08.
    # Boundary analysis_time pad-pick for the init 06 = idx 2 (06), which is
    # launched exactly at init and therefore usable.
    # lead_at_first_target = (07-06)/1d = 1, which already covers
    # num_past=1, so no shift back. Window at target 07: lead 1, [0..2].
    # Window at target 08: lead 2, [1..3].
    expected_analysis_idx = 2
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(
        boundary[0, 0, :], boundary_values[expected_analysis_idx, 0:3]
    )
    np.testing.assert_array_equal(
        boundary[1, 0, :], boundary_values[expected_analysis_idx, 1:4]
    )


def test_forecast_boundary_anchors_on_init_not_target():
    """A boundary forecast launched after model init (between the last
    input state and the first target) must not be selected - operationally
    it would be unavailable. The analysis_time is anchored on the model
    init time, so the latest launch at or before init is used instead."""
    # Interior analysis, 2h step. Sample idx=0 state = 00,02,04,06:
    # model init = 02, first target = 04, second target = 06.
    interior_times = np.datetime64("2020-01-01T00") + np.arange(
        8
    ) * np.timedelta64(2, "h")
    interior_values = np.arange(8, dtype=float)
    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=2),
    )

    # Boundary launches at odd hours (2019-12-31T21, 23, 01, 03, ...),
    # spanning wide enough that no interior cropping is triggered. Launch
    # 01 (idx 2) is the latest <= init (02); launch 03 (idx 3) sits
    # strictly between init (02) and the first target (04). The buggy
    # target-time anchor would pick 03 (a future launch); the fixed
    # init-time anchor picks 01.
    n_analysis = 9
    n_leads = 16
    boundary_analysis = np.datetime64("2019-12-31T21") + np.arange(
        n_analysis
    ) * np.timedelta64(2, "h")
    boundary_leads = np.arange(n_leads) * np.timedelta64(1, "h")
    boundary_values = (
        np.arange(n_analysis).reshape(-1, 1) * 1000
        + np.arange(n_leads).reshape(1, -1) * 10
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(hours=1),
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
    # Launch at 01 = analysis idx 2 (not 03 = idx 3). From 01: target 04
    # is lead (04-01)/1h = 3 -> window [2,5); target 06 is lead 5 ->
    # window [4,7).
    expected_analysis_idx = 2
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(
        boundary[0, 0, :], boundary_values[expected_analysis_idx, 2:5]
    )
    np.testing.assert_array_equal(
        boundary[1, 0, :], boundary_values[expected_analysis_idx, 4:7]
    )


def test_insufficient_boundary_coverage_raises():
    """If the boundary cannot be cropped enough to cover the requested
    past-window, ``get_time_crop_slice`` surfaces a clear error."""
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


def test_forecast_interior_cropped_along_analysis_time():
    """A forecast interior whose earliest launches fall outside the boundary
    coverage is cropped along ``analysis_time`` (whole launches dropped), so
    fewer samples remain and the survivors still build a boundary window."""
    n_analysis = 6
    n_leads = 5
    interior_analysis = np.datetime64("2020-01-01") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    interior_leads = np.arange(n_leads) * np.timedelta64(1, "D")
    interior_values = (
        np.arange(n_analysis).reshape(-1, 1) * 100
        + np.arange(n_leads).reshape(1, -1)
    ).astype(float)
    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=(interior_analysis, interior_leads),
        is_forecast=True,
        step_length=timedelta(days=1),
    )

    # Analysis boundary starts only at 2020-01-04, so the launches at
    # analysis_time 01-01..01-03 have no boundary coverage and are dropped.
    boundary_times = np.datetime64("2020-01-04") + np.arange(
        10
    ) * np.timedelta64(1, "D")
    boundary_values = np.arange(300, 310, dtype=float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=boundary_times,
        is_forecast=False,
        step_length=timedelta(days=1),
    )

    full = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=None,
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    cropped = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    # Launches 01-01..01-03 have no boundary coverage and are dropped; the
    # last launch needs one day of future window beyond its final target.
    assert len(full) == 6
    assert len(cropped) == 2
    _, _, _, boundary, _ = cropped[0]
    # Launch 01-05: targets 01-07 and 01-08, i.e. boundary entries 303 and
    # 304, each with one step of past and future window around it.
    assert boundary.flatten().tolist() == [
        302.0,
        303.0,
        304.0,
        303.0,
        304.0,
        305.0,
    ]


def test_forecast_interior_forcing_cropped_with_state():
    """Cropping the interior against the boundary must move state and forcing
    together: forecast forcing is looked up by positional ``analysis_time``
    index, so cropping only the state would pair every sample with forcing
    from a different launch."""
    n_analysis = 4
    n_leads = 10
    interior_analysis = np.datetime64("2020-01-01") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    interior_leads = np.arange(n_leads) * np.timedelta64(1, "h")
    # Launch k holds states 10k..10k+9 and forcings 100+10k..100+10k+9, so
    # the launch a sample was built from is readable off either tensor.
    state_values = np.arange(n_analysis * n_leads, dtype=float).reshape(
        n_analysis, n_leads
    )
    forcing_values = state_values + 100.0
    interior_datastore = SinglePointDummyDatastore(
        state_data=state_values,
        forcing_data=forcing_values,
        time_values=(interior_analysis, interior_leads),
        is_forecast=True,
        step_length=timedelta(hours=1),
    )

    # Boundary starts three days late, so the first three launches are cropped.
    boundary_times = np.datetime64("2020-01-03") + np.arange(
        170
    ) * np.timedelta64(1, "h")
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=np.arange(1000, 1170, dtype=float),
        time_values=boundary_times,
        is_forecast=False,
        step_length=timedelta(hours=1),
    )

    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
    )

    uncropped = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=None,
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )
    # The crop has to bite, or this would pass vacuously.
    assert 0 < len(dataset) < len(uncropped)

    for idx in range(len(dataset)):
        _, target_states, forcing, _, _ = dataset[idx]
        state_launch = int(target_states.flatten()[0].item()) // 10
        forcing_launch = (int(forcing.flatten()[0].item()) - 100) // 10
        assert state_launch == forcing_launch


def test_forecast_interior_cropped_when_boundary_ends_early():
    """The coverage check must account for the interior forecast's own lead
    times: launches whose rollout runs past the end of an analysis boundary
    are cropped, rather than surviving and failing per-sample."""
    n_analysis = 4
    n_leads = 5
    interior_analysis = np.datetime64("2020-01-05") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    interior_leads = np.arange(n_leads) * np.timedelta64(1, "D")
    interior_values = (
        np.arange(n_analysis).reshape(-1, 1) * 100
        + np.arange(n_leads).reshape(1, -1)
    ).astype(float)
    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=(interior_analysis, interior_leads),
        is_forecast=True,
        step_length=timedelta(days=1),
    )

    # Boundary ends 01-10, but the launch at 01-08 rolls out to 01-10 and
    # needs one further step for the future window.
    boundary_times = np.datetime64("2020-01-03") + np.arange(
        8
    ) * np.timedelta64(1, "D")
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=np.arange(1000, 1008, dtype=float),
        time_values=boundary_times,
        is_forecast=False,
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

    # Launches 01-05 and 01-06 roll out to 01-09 / 01-10, and the latter
    # needs 01-11 for the future window, which the boundary does not have.
    assert len(dataset) == 2
    expected = [
        [1003.0, 1004.0, 1005.0, 1004.0, 1005.0, 1006.0],
        [1004.0, 1005.0, 1006.0, 1005.0, 1006.0, 1007.0],
    ]
    for idx in range(len(dataset)):
        _, _, _, boundary, _ = dataset[idx]
        assert boundary.flatten().tolist() == expected[idx]


def test_forecast_boundary_launch_spacing_differs_from_lead_spacing():
    """A boundary launched 6-hourly with hourly leads must still be windowed
    in lead steps: stepping back one launch buys six window steps, not one."""
    interior_times = np.datetime64("2020-01-02") + np.arange(
        16
    ) * np.timedelta64(3, "h")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(16, dtype=float),
        forcing_data=np.arange(16, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=3),
    )

    boundary_analysis = np.datetime64("2020-01-01") + np.arange(
        16
    ) * np.timedelta64(6, "h")
    boundary_leads = np.arange(25) * np.timedelta64(1, "h")
    boundary_values = (
        np.arange(16).reshape(-1, 1) * 1000 + np.arange(25).reshape(1, -1)
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(hours=6),
    )

    # An 8-step past window exceeds the lead available from the launch that
    # the init time first selects, so the launch has to be stepped back. One
    # launch back buys 6 window steps, so a step-back counted in window steps
    # would overshoot the 24 h horizon.
    dataset = WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=8,
        num_future_boundary_steps=1,
    )

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        _, _, _, boundary, target_times = dataset[idx]
        assert boundary.shape[-1] == 10

        first_target = np.datetime64(int(target_times[0]), "ns")
        model_init = first_target - np.timedelta64(3, "h")
        # The newest launch at or before init that still has 8 lead steps
        # of headroom before the first target.
        expected_launch = max(
            i
            for i, launch in enumerate(boundary_analysis)
            if launch <= model_init
            and (first_target - launch) / np.timedelta64(1, "h") >= 8
        )

        for step in range(boundary.shape[0]):
            window = boundary[step].flatten().tolist()
            # Values encode launch * 1000 + lead, so the window pins both
            # the launch used and that the leads are consecutive.
            assert int(window[0]) // 1000 == expected_launch
            assert window == [window[0] + i for i in range(10)]


def test_short_boundary_forecast_horizon_raises():
    """A boundary whose lead horizon cannot span a whole sample is rejected at
    construction, not once samples are drawn."""
    interior_times = np.datetime64("2020-01-02") + np.arange(
        16
    ) * np.timedelta64(3, "h")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(16, dtype=float),
        forcing_data=np.arange(16, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=3),
    )

    # 12 h of lead cannot cover a 6 h rollout plus a 4 h past window from a
    # launch up to 6 h before init.
    boundary_analysis = np.datetime64("2020-01-01") + np.arange(
        16
    ) * np.timedelta64(6, "h")
    boundary_leads = np.arange(13) * np.timedelta64(1, "h")
    boundary_values = (
        np.arange(16).reshape(-1, 1) * 100 + np.arange(13).reshape(1, -1)
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(hours=6),
    )

    with pytest.raises(ValueError, match="horizon is too short"):
        WeatherDataset(
            datastore=interior_datastore,
            datastore_boundary=boundary_datastore,
            ar_steps=2,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=4,
            num_future_boundary_steps=1,
        )


def test_forecast_boundary_launched_exactly_at_init_is_used():
    """A boundary forecast launched exactly at the model init time is
    available operationally - the interior analysis and the boundary forcing
    both take time to produce, so neither is reliably ready first - and must
    be preferred over the previous, staler launch."""
    interior_times = np.datetime64("2020-01-05") + np.arange(
        8
    ) * np.timedelta64(1, "D")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(8, dtype=float),
        forcing_data=np.arange(8, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(days=1),
    )

    # Launches on every interior day, so the init time 2020-01-06 of sample
    # idx=0 coincides exactly with launch index 2.
    boundary_analysis = np.datetime64("2020-01-04") + np.arange(
        6
    ) * np.timedelta64(1, "D")
    boundary_leads = np.arange(8) * np.timedelta64(1, "D")
    boundary_values = (
        np.arange(6).reshape(-1, 1) * 1000 + np.arange(8).reshape(1, -1) * 10
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
    # Launch index 2 is 2020-01-06, exactly the model init time, and values
    # encode launch * 1000 + lead * 10. This pins both the launch - a
    # strictly-before rule would have picked the staler index 1 - and the
    # leads within it, so a window offset cannot slip through.
    # Targets 01-07 and 01-08 are leads 1 and 2 of that launch.
    assert boundary.flatten().tolist() == [
        2000.0,
        2010.0,
        2020.0,
        2010.0,
        2020.0,
        2030.0,
    ]


def _aligned_boundary_case(n_past, n_future, lead_hours, ar_steps=2):
    """Build a 6-hourly interior against a 6-hourly-launched boundary.

    Interior times and boundary launches share the 00/06/12/18 grid, as an
    operational ERA5 boundary would.

    Parameters
    ----------
    n_past, n_future : int
        Boundary window size.
    lead_hours : sequence of int
        Boundary lead times, in hours.
    ar_steps : int, optional
        Autoregressive steps. Default ``2``.

    Returns
    -------
    WeatherDataset
        The constructed dataset.
    """
    interior_times = np.datetime64("2020-01-02") + np.arange(
        12
    ) * np.timedelta64(6, "h")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(12, dtype=float),
        forcing_data=np.arange(12, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )
    boundary_analysis = np.datetime64("2020-01-01") + np.arange(
        16
    ) * np.timedelta64(6, "h")
    boundary_leads = np.array(
        [np.timedelta64(h, "h") for h in lead_hours], dtype="timedelta64[ns]"
    )
    boundary_values = (
        np.arange(16).reshape(-1, 1) * 1000
        + np.arange(len(boundary_leads)).reshape(1, -1)
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(hours=6),
    )
    return WeatherDataset(
        datastore=interior_datastore,
        datastore_boundary=boundary_datastore,
        ar_steps=ar_steps,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        num_past_boundary_steps=n_past,
        num_future_boundary_steps=n_future,
    )


def test_boundary_horizon_accepts_exactly_sufficient_launch():
    """A launch grid aligned with the interior needs no slack: with a
    2-step rollout and no window padding, a 12 h horizon is exactly enough
    and must be accepted, not charged a spurious extra launch offset."""
    dataset = _aligned_boundary_case(
        n_past=0, n_future=0, lead_hours=[0, 6, 12]
    )

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        _, _, _, boundary, _ = dataset[idx]
        assert boundary.shape[-1] == 1
        # Every window comes from the launch at the sample's init time.
        assert np.all(boundary.numpy() // 1000 == boundary.numpy()[0] // 1000)


def test_boundary_horizon_rejects_one_step_short():
    """One lead step less than the accepted case must be refused at
    construction, so the margin is pinned from both sides."""
    with pytest.raises(ValueError, match="horizon is too short"):
        _aligned_boundary_case(n_past=0, n_future=0, lead_hours=[0, 6])


def test_boundary_horizon_accounts_for_launch_grid_alignment():
    """When the interior steps finer than the boundary launch spacing, the
    chosen launch can sit almost a full launch interval before init, and the
    horizon check has to account for that rather than estimating from step
    lengths alone."""
    interior_times = np.datetime64("2020-01-02") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(24, dtype=float),
        forcing_data=np.arange(24, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=1),
    )
    boundary_analysis = np.datetime64("2020-01-01") + np.arange(
        16
    ) * np.timedelta64(6, "h")
    boundary_leads = np.arange(9) * np.timedelta64(1, "h")
    boundary_values = (
        np.arange(16).reshape(-1, 1) * 1000 + np.arange(9).reshape(1, -1)
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
        step_length=timedelta(hours=6),
    )

    with pytest.raises(ValueError, match="horizon is too short"):
        WeatherDataset(
            datastore=interior_datastore,
            datastore_boundary=boundary_datastore,
            ar_steps=2,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=3,
            num_future_boundary_steps=0,
        )


def test_forecast_interior_with_forecast_boundary():
    """Forecast interior + forecast boundary, the fourth analysis/forecast
    combination.

    This is the only pairing where the interior's init time differs from its
    `analysis_time`, so it is the only one that can catch a regression to
    anchoring the boundary launch on the launch time rather than on the init
    time. It is also the only path that reaches the forecast branches of
    `_state_time_step` and `_max_state_lead_used`.
    """
    n_analysis = 5
    n_leads = 6
    interior_analysis = np.datetime64("2020-01-05") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    interior_leads = np.arange(n_leads) * np.timedelta64(1, "D")
    interior_values = (
        np.arange(n_analysis).reshape(-1, 1) * 100
        + np.arange(n_leads).reshape(1, -1)
    ).astype(float)
    interior_datastore = SinglePointDummyDatastore(
        state_data=interior_values,
        forcing_data=interior_values,
        time_values=(interior_analysis, interior_leads),
        is_forecast=True,
        step_length=timedelta(days=1),
    )

    boundary_analysis = np.datetime64("2020-01-03") + np.arange(
        8
    ) * np.timedelta64(1, "D")
    boundary_leads = np.arange(10) * np.timedelta64(1, "D")
    boundary_values = (
        np.arange(8).reshape(-1, 1) * 1000 + np.arange(10).reshape(1, -1) * 10
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

    assert len(dataset) == n_analysis
    for idx in range(len(dataset)):
        _, target_states, forcing, boundary, _ = [
            t.numpy() for t in dataset[idx]
        ]
        # State and forcing come from the same interior launch and, with no
        # forcing window, the same lead times.
        assert target_states.flatten().tolist() == forcing.flatten().tolist()
        assert np.all(target_states.flatten() // 100 == idx)

        # Interior launch idx is 2020-01-05+idx, so init is its lead 1,
        # 2020-01-06+idx. Anchoring on the launch instead would pick the
        # boundary launched a day earlier.
        model_init = interior_analysis[idx] + interior_leads[1]
        expected_boundary_launch = int(
            np.searchsorted(boundary_analysis, model_init, side="right") - 1
        )
        assert np.all(boundary.flatten() // 1000 == expected_boundary_launch)


def _boundary_with_analysis_times(analysis_times):
    """Build a forecast boundary datastore on the given launch times.

    Parameters
    ----------
    analysis_times : np.ndarray
        Launch times, which may deliberately repeat or be unsorted.

    Returns
    -------
    BoundaryOnlyDummyDatastore
        The boundary datastore.
    """
    leads = np.arange(12) * np.timedelta64(1, "h")
    values = (
        np.arange(len(analysis_times)).reshape(-1, 1) * 1000
        + np.arange(len(leads)).reshape(1, -1)
    ).astype(float)
    return BoundaryOnlyDummyDatastore(
        forcing_data=values,
        time_values=(np.array(analysis_times), leads),
        is_forecast=True,
        step_length=timedelta(hours=6),
    )


def _interior_for_boundary_checks():
    """Build a 6-hourly analysis interior for boundary-axis checks.

    Returns
    -------
    SinglePointDummyDatastore
        The interior datastore.
    """
    times = np.datetime64("2020-01-02") + np.arange(8) * np.timedelta64(6, "h")
    return SinglePointDummyDatastore(
        state_data=np.arange(8, dtype=float),
        forcing_data=np.arange(8, dtype=float),
        time_values=times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )


@pytest.mark.parametrize(
    "analysis_times,match",
    [
        # npyfilesmeps repeats each launch once per ensemble member
        (
            np.repeat(
                np.datetime64("2020-01-01")
                + np.arange(8) * np.timedelta64(6, "h"),
                2,
            ),
            "must be unique",
        ),
        (
            (
                np.datetime64("2020-01-01")
                + np.arange(8) * np.timedelta64(6, "h")
            )[::-1],
            "must be sorted",
        ),
    ],
)
def test_boundary_analysis_times_must_support_pad_lookup(analysis_times, match):
    """Launches are located with a `pad` lookup, which needs a unique sorted
    index. A duplicated or unsorted axis must be named rather than surfacing
    as a bare pandas `InvalidIndexError`."""
    with pytest.raises(ValueError, match=match):
        WeatherDataset(
            datastore=_interior_for_boundary_checks(),
            datastore_boundary=_boundary_with_analysis_times(analysis_times),
            ar_steps=2,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=0,
            num_future_boundary_steps=0,
        )


def test_too_few_samples_error_names_the_boundary_window():
    """When a boundary datastore is configured, the interior is cropped to
    the boundary coverage, so the remedy list has to name the boundary window
    and not send the user after `ar_steps` and the forcing window alone."""
    interior_times = np.datetime64("2020-01-02") + np.arange(
        6
    ) * np.timedelta64(6, "h")
    interior = SinglePointDummyDatastore(
        state_data=np.arange(6, dtype=float),
        forcing_data=np.arange(6, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )
    boundary_times = np.datetime64("2020-01-02") + np.arange(
        6
    ) * np.timedelta64(6, "h")
    boundary = BoundaryOnlyDummyDatastore(
        forcing_data=np.arange(6, dtype=float),
        time_values=boundary_times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )

    with pytest.raises(ValueError, match="num_past_boundary_steps"):
        WeatherDataset(
            datastore=interior,
            datastore_boundary=boundary,
            ar_steps=3,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=1,
            num_future_boundary_steps=1,
        )
