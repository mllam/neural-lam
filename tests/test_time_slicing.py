# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
import numpy as np
import pytest
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore
from neural_lam.utils.time import get_time_step
from neural_lam.weather_dataset import WeatherDataset


class SinglePointDummyDatastore(BaseDatastore):
    """One-grid-point datastore. In forecast mode ``time_values`` is the pair
    ``(analysis_times, elapsed_forecast_durations)`` and the data is 2D."""

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
            # add `{category}_feature` and `grid_index` dimensions

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
    """Boundary-only variant providing forcing but no state."""

    def __init__(
        self,
        time_values,
        forcing_data,
        is_forecast=False,
        step_length=timedelta(hours=1),
    ):
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
    assert len(sample) == 5


def _interior_times():
    return np.datetime64("2020-01-01") + np.arange(len(ANALYSIS_STATE_VALUES))


def _boundary_times_aligned():
    """Boundary times padded by BOUNDARY_PAD steps on both sides of the
    interior, so no cropping is needed."""
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
    ],
)
def test_time_slicing_boundary_analysis(
    ar_steps, num_past_boundary_steps, num_future_boundary_steps
):
    """Boundary windows for an analysis interior and analysis boundary."""
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

    # The first target is t_2, i.e. boundary index BOUNDARY_PAD + 2
    boundary_center = BOUNDARY_PAD + 2
    window_size = num_past_boundary_steps + num_future_boundary_steps + 1
    assert boundary.shape == (ar_steps, 1, window_size)
    for i in range(ar_steps):
        start = boundary_center + i - num_past_boundary_steps
        end = boundary_center + i + num_future_boundary_steps + 1
        expected = BOUNDARY_FORCING_VALUES[start:end]
        np.testing.assert_array_equal(boundary[i, 0, :], expected)


def test_boundary_step_length_mismatch_supported():
    """A 6h boundary is windowed around the times of a 1h interior."""
    interior_times = np.datetime64("2020-01-01") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    interior_values = np.arange(24, dtype=float)

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
    """Forecast interior with an analysis boundary."""
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
    """Analysis interior with a forecast boundary, including a boundary
    launched exactly at the model init time."""
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
    # Init is 01-06, targets 01-07 and 01-08. The launch at 01-06 (index
    # 2) is used, so the targets are its leads 1 and 2.
    expected_analysis_idx = 2
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(
        boundary[0, 0, :], boundary_values[expected_analysis_idx, 0:3]
    )
    np.testing.assert_array_equal(
        boundary[1, 0, :], boundary_values[expected_analysis_idx, 1:4]
    )


def test_forecast_boundary_anchors_on_init_not_target():
    """A boundary launched after the model init time is not used, even if
    it is before the first target."""
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

    # Launches at odd hours: 01 (index 2) is the latest at or before init
    # (02), 03 (index 3) lies between init and the first target (04).
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
    # From launch 01, targets 04 and 06 are leads 3 and 5
    expected_analysis_idx = 2
    assert boundary.shape == (2, 1, 3)
    np.testing.assert_array_equal(
        boundary[0, 0, :], boundary_values[expected_analysis_idx, 2:5]
    )
    np.testing.assert_array_equal(
        boundary[1, 0, :], boundary_values[expected_analysis_idx, 4:7]
    )


def test_insufficient_boundary_coverage_raises():
    """A boundary window no sample can be covered for is rejected."""
    interior_datastore = SinglePointDummyDatastore(
        state_data=ANALYSIS_STATE_VALUES,
        forcing_data=FORCING_VALUES,
        time_values=_interior_times(),
        is_forecast=False,
    )
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=BOUNDARY_FORCING_VALUES[:10],
        time_values=_interior_times(),
        is_forecast=False,
    )

    with pytest.raises(ValueError, match="cannot align"):
        WeatherDataset(
            datastore=interior_datastore,
            datastore_boundary=boundary_datastore,
            ar_steps=3,
            num_past_forcing_steps=0,
            num_future_forcing_steps=0,
            num_past_boundary_steps=20,
            num_future_boundary_steps=20,
        )


def test_forecast_interior_forcing_cropped_with_state():
    """Cropping a forecast interior to the boundary coverage crops state and
    forcing together, so each sample pairs them from the same launch."""
    n_analysis = 4
    n_leads = 10
    interior_analysis = np.datetime64("2020-01-01") + np.arange(
        n_analysis
    ) * np.timedelta64(1, "D")
    interior_leads = np.arange(n_leads) * np.timedelta64(1, "h")
    # Launch k holds states 10k..10k+9 and forcings 100 higher
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
    assert 0 < len(dataset) < len(uncropped)

    for idx in range(len(dataset)):
        _, target_states, forcing, _, _ = dataset[idx]
        state_launch = int(target_states.flatten()[0].item()) // 10
        forcing_launch = (int(forcing.flatten()[0].item()) - 100) // 10
        assert state_launch == forcing_launch


def test_forecast_interior_cropped_when_boundary_ends_early():
    """Launches whose rollout runs past the end of the boundary are
    cropped."""
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

    assert len(dataset) == 2
    expected = [
        [1003.0, 1004.0, 1005.0, 1004.0, 1005.0, 1006.0],
        [1004.0, 1005.0, 1006.0, 1005.0, 1006.0, 1007.0],
    ]
    for idx in range(len(dataset)):
        _, _, _, boundary, _ = dataset[idx]
        assert boundary.flatten().tolist() == expected[idx]


def _aligned_boundary_case(n_past, n_future, lead_hours, ar_steps=2):
    """Build a dataset with a 6-hourly interior and a forecast boundary
    launched on the same 6-hourly grid."""
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
    """A boundary horizon that exactly covers a 2-step rollout is
    accepted."""
    dataset = _aligned_boundary_case(
        n_past=0, n_future=0, lead_hours=[0, 6, 12]
    )
    assert len(dataset) > 0


def test_boundary_horizon_rejects_one_step_short():
    """One lead step less than the accepted case is rejected."""
    with pytest.raises(ValueError, match="horizon is too short"):
        _aligned_boundary_case(n_past=0, n_future=0, lead_hours=[0, 6])


def test_forecast_interior_with_forecast_boundary():
    """Forecast interior with a forecast boundary, where the boundary
    launch is chosen from the init time, not the interior launch time."""
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
        assert target_states.flatten().tolist() == forcing.flatten().tolist()
        assert np.all(target_states.flatten() // 100 == idx)

        # Init is lead 1 of interior launch idx
        model_init = interior_analysis[idx] + interior_leads[1]
        expected_boundary_launch = int(
            np.searchsorted(boundary_analysis, model_init, side="right") - 1
        )
        # Targets are leads 1 and 2 of that boundary launch
        base = expected_boundary_launch * 1000
        assert boundary.flatten().tolist() == [
            base + 0.0,
            base + 10.0,
            base + 20.0,
            base + 10.0,
            base + 20.0,
            base + 30.0,
        ]


def _boundary_with_analysis_times(analysis_times):
    """Build a forecast boundary datastore on the given launch times."""
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
    """Build a 6-hourly analysis interior."""
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
    """The boundary `analysis_time` has to be unique and sorted."""
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


def test_forecast_boundary_leads_not_starting_at_zero():
    """Boundary leads that start after zero are indexed from the first
    lead."""
    interior_times = np.datetime64("2020-01-03") + np.arange(
        10
    ) * np.timedelta64(6, "h")
    interior_datastore = SinglePointDummyDatastore(
        state_data=np.arange(10, dtype=float),
        forcing_data=np.arange(10, dtype=float),
        time_values=interior_times,
        is_forecast=False,
        step_length=timedelta(hours=6),
    )

    # Leads start at +12 h
    boundary_analysis = np.datetime64("2020-01-01") + np.arange(
        12
    ) * np.timedelta64(6, "h")
    boundary_leads = np.timedelta64(12, "h") + np.arange(10) * np.timedelta64(
        6, "h"
    )
    boundary_values = (
        np.arange(12).reshape(-1, 1) * 1000 + np.arange(10).reshape(1, -1)
    ).astype(float)
    boundary_datastore = BoundaryOnlyDummyDatastore(
        forcing_data=boundary_values,
        time_values=(boundary_analysis, boundary_leads),
        is_forecast=True,
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

    assert len(dataset) > 0
    lead_step = np.timedelta64(6, "h")
    for idx in range(len(dataset)):
        _, _, _, boundary, target_times = dataset[idx]
        window = boundary.flatten().tolist()
        assert len(window) == 2 * 3

        first_target = np.datetime64(int(target_times[0]), "ns")
        model_init = first_target - lead_step
        bound = min(model_init, first_target - boundary_leads[0] - lead_step)
        launch = int(np.searchsorted(boundary_analysis, bound, "right") - 1)

        # Values encode launch * 1000 + lead position from leads[0]
        expected = []
        for step in range(2):
            target = np.datetime64(int(target_times[step]), "ns")
            lead = int(
                (target - boundary_analysis[launch] - boundary_leads[0])
                / lead_step
            )
            expected += [
                launch * 1000 + lead - 1.0,
                launch * 1000 + lead + 0.0,
                launch * 1000 + lead + 1.0,
            ]
        assert window == expected


def test_get_time_step_rejects_irregular_spacing():
    """A gap in the time axis is rejected."""
    times = np.datetime64("2020-01-01") + np.array([0, 6, 15], dtype="m8[h]")
    with pytest.raises(ValueError, match="Inconsistent time steps"):
        get_time_step(times)
