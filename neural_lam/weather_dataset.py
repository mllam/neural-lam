"""Dataset helpers wrapping Neural-LAM datastores for PyTorch Lightning."""

# Standard library
import datetime
import warnings
from typing import Any, Iterator, cast

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore
from neural_lam.utils import (
    apply_time_crop,
    get_integer_time,
    get_time_crop_slice,
    get_time_step,
)


def _format_timedelta(td: np.timedelta64) -> str:
    """Render a timedelta in its largest exact unit, e.g. ``"13 hours"``.

    Parameters
    ----------
    td : np.timedelta64
        The interval to render.

    Returns
    -------
    str
        Value and unit.
    """
    microseconds = int(td / np.timedelta64(1, "us"))
    value, unit = get_integer_time(
        datetime.timedelta(microseconds=microseconds)
    )
    if unit == "unknown":
        return str(td)
    return f"{value} {unit}"


def _latest_usable_launch(
    model_init_time: np.datetime64,
    first_target_time: np.datetime64,
    lead_offset: np.timedelta64,
    lead_step: np.timedelta64,
    num_past_steps: int,
) -> np.datetime64:
    """Upper bound on the boundary launch usable for one sample.

    A launch is usable when it starts at or before the model init time (a
    later one would be unavailable operationally) and still leaves
    ``num_past_steps`` of lead before the first target. Lead position falls
    monotonically as the launch gets later, so both conditions collapse to a
    single upper bound and the latest usable launch is the one that
    ``pad``-matches it.

    Parameters
    ----------
    model_init_time, first_target_time : np.datetime64
        Init time and first target time of the sample.
    lead_offset : np.timedelta64
        First ``elapsed_forecast_duration`` of the boundary forecast, which
        need not be zero.
    lead_step : np.timedelta64
        Spacing between boundary lead times.
    num_past_steps : int
        Past window size, in lead steps.

    Returns
    -------
    np.datetime64
        The latest launch time that could serve this sample.
    """
    past_window_bound = (
        first_target_time - lead_offset - num_past_steps * lead_step
    )
    return min(model_init_time, past_window_bound)


def _check_window_bounds(
    window_start: int,
    window_end: int,
    axis_size: int,
    target_time: np.datetime64,
    num_past_steps: int,
    num_future_steps: int,
    dim_name: str,
) -> None:
    """Raise if a forcing/boundary window runs off either end of its axis.

    ``xr.DataArray.isel`` wraps a negative slice start and truncates a slice
    end past the array, which would otherwise only surface as an opaque
    coordinate-size error further down.

    Raises
    ------
    ValueError
        If the window is not fully contained in ``[0, axis_size)``.
    """
    if window_start < 0 or window_end > axis_size:
        raise ValueError(
            f"Forcing/boundary does not cover the window "
            f"[{window_start}, {window_end}) along `{dim_name}` (axis size "
            f"{axis_size}) around target time {target_time}: "
            f"{num_past_steps} steps before and {num_future_steps} steps "
            "after the target are required."
        )


class WeatherDataset(torch.utils.data.Dataset):
    """Dataset class for weather data.

    Loads and processes weather data from a given datastore, with optional
    boundary forcing from a separate boundary datastore. Boundary windowing
    is aligned to interior state times by nearest-neighbor lookup, so the
    two datastores may differ in step length and either may be analysis or
    forecast data.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore to load the data from (e.g. mdp).
    split : str, optional
        The data split to use ("train", "val" or "test"). Default is "train".
    ar_steps : int, optional
        The number of autoregressive steps. Default is 3.
    num_past_forcing_steps: int, optional
        Number of past time steps to include in forcing input. If set to i,
        forcing from times t-i, t-i+1, ..., t-1, t (and potentially beyond,
        given num_future_forcing_steps) are included as forcing inputs at time t
        Default is 1.
    num_future_forcing_steps: int, optional
        Number of future time steps to include in forcing input. If set to j,
        forcing from times t, t+1, ..., t+j-1, t+j (and potentially times before
        t, given num_past_forcing_steps) are included as forcing inputs at time
        t. Default is 1.
    num_past_boundary_steps: int, optional
        Number of past time steps to include in boundary forcing input.
        Default is 1.
    num_future_boundary_steps: int, optional
        Number of future time steps to include in boundary forcing input.
        Default is 1.
    datastore_boundary : BaseDatastore, optional
        Separate datastore providing boundary forcing. If None, the boundary
        tensor is empty.
    load_single_member : bool, optional
        If `False` and the datastore returns an ensemble of state
        realisations, treat each state ensemble member as an independent
        sample. If `True`, only ensemble member 0 is used. Default is False,
        so all members are used when available.
    """

    INIT_STEPS = 2

    def __init__(
        self,
        datastore: BaseDatastore,
        split: str = "train",
        ar_steps: int = 3,
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        num_past_boundary_steps: int = 1,
        num_future_boundary_steps: int = 1,
        datastore_boundary: BaseDatastore | None = None,
        load_single_member: bool = False,
    ) -> None:
        """
        Construct a ``WeatherDataset``. See the class docstring for the
        constructor parameters.

        Raises
        ------
        ValueError
            If the datastore does not provide state data, if the configured
            ``ar_steps`` and forcing windows leave zero samples in ``split``,
            or if the state/forcing dimension order does not match the
            datastore's expected dimension order.
        """
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.datastore = datastore
        self.datastore_boundary = datastore_boundary
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.load_single_member = load_single_member

        da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        if da_state is None:
            raise ValueError(
                "The datastore must provide state data for the WeatherDataset."
            )
        self.da_state = da_state
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )

        if self.datastore_boundary is not None:
            self.da_boundary_forcing = self.datastore_boundary.get_dataarray(
                category="forcing", split=self.split
            )
        else:
            self.da_boundary_forcing = None

        self._forecast_step_boundary = None
        if self.datastore_boundary is not None:
            datastore_boundary = self.datastore_boundary
            if (
                self.da_boundary_forcing is not None
                and datastore_boundary.is_forecast
            ):
                self._forecast_step_boundary = get_time_step(
                    self.da_boundary_forcing.elapsed_forecast_duration.values
                )

            # Forecast forcing is looked up by positional `analysis_time`
            # index, so state and forcing must be cropped by the same slice
            # or samples would pair them from different launches.
            if self.da_boundary_forcing is not None:
                crop_dim, crop_slice = get_time_crop_slice(
                    self.da_state,
                    self.da_boundary_forcing,
                    da_requested_is_forecast=self.datastore.is_forecast,
                    da_available_is_forecast=datastore_boundary.is_forecast,
                    num_past_steps=self.num_past_boundary_steps,
                    num_future_steps=self.num_future_boundary_steps,
                    requested_max_lead=self._max_state_lead_used(),
                )
                self.da_state = apply_time_crop(
                    self.da_state, crop_dim, crop_slice
                )
                if self.da_forcing is not None and self.datastore.is_forecast:
                    self.da_forcing = apply_time_crop(
                        self.da_forcing, crop_dim, crop_slice
                    )
                if datastore_boundary.is_forecast:
                    self._check_boundary_analysis_times()
                    self._check_boundary_forecast_horizon()

        if self.datastore.is_ensemble and self.load_single_member:
            warnings.warn(
                "only using first ensemble member, so dataset size is "
                "effectively reduced by the number of ensemble members "
                f"({self.da_state.ensemble_member.size})",
                UserWarning,
                stacklevel=2,
            )

        # check that with the provided data-arrays and ar_steps that we have a
        # non-zero amount of samples
        if self.__len__() <= 0 and self.da_state is not None:
            remedies = (
                "the number of autoregressive steps (`ar_steps`) and/or the "
                "forcing window size (`num_past_forcing_steps` and "
                "`num_future_forcing_steps`)"
            )
            if self.datastore_boundary is not None:
                remedies += (
                    ", or the boundary window (`num_past_boundary_steps` and "
                    "`num_future_boundary_steps`), which determines how much "
                    "of the interior is cropped to stay within the boundary "
                    "coverage"
                )
            raise ValueError(
                "The provided datastore only provides "
                f"{len(self.da_state.time)} total time steps, which is too few "
                "to create a single sample for the WeatherDataset "
                f"configuration used in the `{split}` split. You could try "
                f"reducing {remedies}."
            )

        # Check the dimensions and their ordering
        parts = dict(state=self.da_state)
        if self.da_forcing is not None:
            parts["forcing"] = self.da_forcing

        for part, da in parts.items():
            if da is not None:
                expected_dim_order = self.datastore.expected_dim_order(
                    category=part
                )
                if da.dims != expected_dim_order:
                    raise ValueError(
                        f"The dimension order of the `{part}` data ({da.dims}) "
                        f"does not match the expected dimension order "
                        f"({expected_dim_order}). Maybe you forgot to "
                        "transpose the data in `BaseDatastore.get_dataarray`?"
                    )

    def __len__(self) -> int:
        """
        Return the number of autoregressive training samples available.

        Returns
        -------
        int
            Number of (init, target) pairs derivable from the datastore.
        """
        assert self.da_state is not None
        if self.datastore.is_forecast:
            # for now we simply create a single sample for each analysis time
            # and then take the first (2 + ar_steps) forecast times.
            # If the datastore returns an ensemble of state realisations and
            # `load_single_member=False`, each ensemble member is exposed as an
            # independent sample by scaling the base dataset length below.

            # Check that there are enough forecast steps available to create
            # samples. The required minimum is the larger of 2 (for the two
            # initial states) and num_past_forcing_steps, plus ar_steps.
            n_forecast_steps = self.da_state.elapsed_forecast_duration.size
            required_state_steps = (
                max(2, self.num_past_forcing_steps) + self.ar_steps
            )
            if n_forecast_steps < required_state_steps:
                raise ValueError(
                    "The number of forecast steps available "
                    f"({n_forecast_steps}) is less than the required "
                    f"{required_state_steps} (max(2, "
                    f"num_past_forcing_steps={self.num_past_forcing_steps})"
                    f" + ar_steps={self.ar_steps}) for creating a sample "
                    "with initial and target states."
                )

            if self.da_forcing is not None:
                # When forcing is present, the forecast horizon must also
                # cover num_future_forcing_steps beyond the last target step.
                n_forcing_forecast_steps = (
                    self.da_forcing.elapsed_forecast_duration.size
                )
                required_forcing_steps = (
                    required_state_steps + self.num_future_forcing_steps
                )
                if n_forcing_forecast_steps < required_forcing_steps:
                    raise ValueError(
                        "The number of forcing forecast steps available "
                        f"({n_forcing_forecast_steps}) is less than the "
                        f"required {required_forcing_steps} "
                        f"(max(2, num_past_forcing_steps="
                        f"{self.num_past_forcing_steps}) + ar_steps="
                        f"{self.ar_steps} + num_future_forcing_steps="
                        f"{self.num_future_forcing_steps}) for "
                        "constructing forcing windows."
                    )

            base_len = self.da_state.analysis_time.size
        else:
            # Number of valid sample start indices in a contiguous time
            # series. With T total time steps and a per-sample window of
            # W = max(2, num_past_forcing_steps) + ar_steps +
            # num_future_forcing_steps, valid start indices are
            # [0 .. T - W], i.e. (T - W + 1) samples in total.
            window = (
                max(2, self.num_past_forcing_steps)
                + self.ar_steps
                + self.num_future_forcing_steps
            )
            n_state_samples = len(self.da_state.time) - window + 1
            if self.da_forcing is not None:
                n_forcing_samples = len(self.da_forcing.time) - window + 1
                base_len = max(0, min(n_state_samples, n_forcing_samples))
            else:
                base_len = max(0, n_state_samples)
        if self.datastore.is_ensemble and not self.load_single_member:
            return base_len * self.da_state.ensemble_member.size
        return base_len

    def _state_time_step(self) -> np.timedelta64:
        """Spacing between consecutive state times within one sample.

        Returns
        -------
        np.timedelta64
            Lead-time spacing for a forecast interior, ``time`` spacing
            otherwise.
        """
        assert self.da_state is not None
        if self.datastore.is_forecast:
            return get_time_step(self.da_state.elapsed_forecast_duration.values)
        return get_time_step(self.da_state.time.values)

    def _sample_window_times(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Init, first-target and last-target time of every sample.

        Returns
        -------
        tuple of np.ndarray
            Three equal-length arrays of datetime64, one entry per sample
            (before any ensemble-member expansion). Empty when the split is
            too short to yield a sample.
        """
        assert self.da_state is not None
        init_steps = self.INIT_STEPS
        offset = max(0, self.num_past_forcing_steps - init_steps)
        n_total = init_steps + self.ar_steps

        if self.datastore.is_forecast:
            leads = self.da_state.elapsed_forecast_duration.values
            if len(leads) < offset + n_total:
                return (np.array([]), np.array([]), np.array([]))
            launches = self.da_state.analysis_time.values
            return (
                launches + leads[offset + init_steps - 1],
                launches + leads[offset + init_steps],
                launches + leads[offset + n_total - 1],
            )

        times = self.da_state.time.values
        # `__len__` also drops the trailing `num_future_forcing_steps`
        # samples and honours a shorter forcing axis. Using the state-only
        # bound here would validate samples the dataset never yields, and
        # their later targets would demand a longer boundary horizon than
        # any real sample needs.
        n_samples = len(times) - offset - n_total + 1
        n_samples -= self.num_future_forcing_steps
        if self.da_forcing is not None:
            n_samples = min(
                n_samples,
                len(self.da_forcing.time)
                - offset
                - n_total
                + 1
                - self.num_future_forcing_steps,
            )
        if n_samples <= 0:
            return (np.array([]), np.array([]), np.array([]))
        first = offset + init_steps - 1
        return (
            times[first : first + n_samples],
            times[first + 1 : first + 1 + n_samples],
            times[first + self.ar_steps : first + self.ar_steps + n_samples],
        )

    def _check_boundary_analysis_times(self) -> None:
        """Check the boundary launch axis supports a `pad` lookup.

        Launches are located with `Index.get_indexer(method="pad")`, which
        needs a unique, sorted index; npyfilesmeps in particular repeats
        each analysis time once per ensemble member. Without this the user
        gets a bare pandas message naming neither the datastore nor the
        coordinate.

        Raises
        ------
        ValueError
            If `analysis_time` has duplicates or is unsorted.
        """
        assert self.da_boundary_forcing is not None
        index = self.da_boundary_forcing.get_index("analysis_time")

        if not index.is_unique:
            duplicates = index[index.duplicated()].unique().tolist()
            raise ValueError(
                "The boundary datastore's `analysis_time` must be unique, "
                f"but {len(duplicates)} value(s) repeat, e.g. "
                f"{duplicates[:3]}. A boundary datastore that repeats each "
                "launch per ensemble member has to be de-duplicated first."
            )
        if not index.is_monotonic_increasing:
            raise ValueError(
                "The boundary datastore's `analysis_time` must be sorted in "
                "increasing order."
            )

    def _check_boundary_forecast_horizon(self) -> None:
        """Check every sample's window fits inside one boundary launch.

        Each sample is windowed from a single launch, so that launch needs
        lead time out to the last target plus the future window. Cropping
        the interior cannot fix a horizon that is too short, so this is
        checked up front rather than failing per-sample mid-epoch. The
        launch is resolved with the same rule the windowing uses, so the
        check is exact rather than an estimate from step lengths.

        Raises
        ------
        ValueError
            If no launch is early enough for a sample, or if the boundary
            forecast horizon cannot span a sample's window.
        """
        assert self.da_boundary_forcing is not None
        init_times, first_targets, last_targets = self._sample_window_times()
        if len(init_times) == 0:
            return

        leads = self.da_boundary_forcing.elapsed_forecast_duration.values
        lead_step = get_time_step(leads)
        analysis_index = self.da_boundary_forcing.analysis_time.get_index(
            "analysis_time"
        )

        launch_bounds = np.array(
            [
                _latest_usable_launch(
                    init_time,
                    first_target,
                    leads[0],
                    lead_step,
                    self.num_past_boundary_steps,
                )
                for init_time, first_target in zip(init_times, first_targets)
            ]
        )
        launch_idx = analysis_index.get_indexer(launch_bounds, method="pad")
        if (launch_idx < 0).any():
            earliest = init_times[launch_idx < 0].min()
            raise ValueError(
                "No boundary forecast is launched early enough for the model "
                f"init time {earliest} with "
                f"{self.num_past_boundary_steps} past window steps. The "
                "boundary datastore must start earlier, or use a smaller "
                "`num_past_boundary_steps`."
            )

        launches = analysis_index.values[launch_idx]
        last_lead = np.floor(
            (last_targets - launches - leads[0]) / lead_step
        ).astype(int)
        needed = int(last_lead.max()) + self.num_future_boundary_steps
        if needed > len(leads) - 1:
            required = leads[0] + needed * lead_step
            raise ValueError(
                "The boundary forecast horizon is too short: each sample is "
                "windowed from a single boundary launch, which here needs "
                f"lead time out to {_format_timedelta(required)} to cover "
                f"{self.ar_steps} autoregressive steps with a "
                f"({self.num_past_boundary_steps}, "
                f"{self.num_future_boundary_steps}) window, but the boundary "
                f"forecasts only run out to {_format_timedelta(leads[-1])}. "
                "Reduce `ar_steps` or the boundary window, or use boundary "
                "forecasts with a longer horizon."
            )

    def _max_state_lead_used(self) -> np.timedelta64 | None:
        """Largest state lead time read per sample, for forecast interiors.

        ``_slice_state_time`` walks only ``INIT_STEPS + ar_steps`` leads, so
        demanding boundary coverage out to the full forecast length would
        crop away usable launches.

        Returns
        -------
        np.timedelta64 or None
            Lead time of the last state step read, or ``None`` for an
            analysis interior.
        """
        if not self.datastore.is_forecast:
            return None
        assert self.da_state is not None
        leads = self.da_state.elapsed_forecast_duration.values
        offset = max(0, self.num_past_forcing_steps - self.INIT_STEPS)
        last_idx = offset + self.INIT_STEPS + self.ar_steps - 1
        return leads[min(last_idx, len(leads) - 1)]

    def _slice_state_time(
        self, da_state: xr.DataArray, idx: int, n_steps: int
    ) -> xr.DataArray:
        """Slice ``da_state`` by integer ``idx`` into one training sample.

        For analysis data the sample's ``time`` is contiguous; for forecast
        data we pick a single ``analysis_time`` and walk its lead times.
        The leading offset accounts for ``num_past_forcing_steps`` so the
        forcing window of the very first sample is in-bounds.

        Returns
        -------
        da_sliced : xr.DataArray
            Sliced state with a single ``time`` dimension covering
            ``INIT_STEPS + n_steps`` consecutive state times.
        """
        init_steps = self.INIT_STEPS
        n_total = init_steps + n_steps
        offset = max(0, self.num_past_forcing_steps - init_steps)

        if self.datastore.is_forecast:
            da_sliced = da_state.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(offset, offset + n_total),
            )
            da_sliced["time"] = (
                da_sliced.analysis_time + da_sliced.elapsed_forecast_duration
            )
            da_sliced = da_sliced.swap_dims(
                {"elapsed_forecast_duration": "time"}
            )
        else:
            start_idx = idx + offset
            da_sliced = da_state.isel(
                time=slice(start_idx, start_idx + n_total)
            )
        return da_sliced

    def _window_same_forecast_by_idx(
        self,
        da_forcing: xr.DataArray,
        idx: int,
        state_times: xr.DataArray,
        num_past_steps: int,
        num_future_steps: int,
    ) -> xr.DataArray:
        """Window forcing from the same forecast datastore as state.

        Uses integer ``analysis_time=idx`` indexing so it tolerates
        repeated analysis_time values (e.g. npyfilesmeps duplicates the
        analysis_time series). Walks lead times in lockstep with the
        state slice; each window is centered on the corresponding target
        state time.
        """
        init_steps = self.INIT_STEPS
        offset = max(0, self.num_past_forcing_steps - init_steps) + init_steps
        da_list = []
        for step in range(self.ar_steps):
            start_lead = offset + step - num_past_steps
            end_lead = offset + step + num_future_steps + 1
            target_time = state_times[init_steps + step].values

            da_sliced = da_forcing.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(start_lead, end_lead),
            ).rename({"elapsed_forecast_duration": "window"})
            da_sliced = da_sliced.assign_coords(
                window=np.arange(-num_past_steps, num_future_steps + 1)
            )
            da_sliced = da_sliced.expand_dims(dim={"time": [target_time]})
            da_list.append(da_sliced)
        return xr.concat(da_list, dim="time")

    def _window_forcing_in_time(
        self,
        da_forcing: xr.DataArray,
        state_times: xr.DataArray,
        num_past_steps: int,
        num_future_steps: int,
        forecast_step: np.timedelta64 | None,
    ) -> xr.DataArray:
        """Window forcing/boundary in time, aligned to interior state times.

        ``state_times`` is the 1D ``time`` coordinate of the already-sliced
        state sample. For each AR target step the matching forcing time is
        picked by nearest-neighbor ``pad`` lookup (latest forcing time
        ``<=`` state time), and a window of
        ``num_past_steps + num_future_steps + 1`` consecutive forcing
        entries is taken around it.

        When ``da_forcing`` has an ``analysis_time`` dimension a single
        launch is resolved for the sample (see
        :func:`_latest_usable_launch`) and the windows are walked across its
        lead times.

        Returns
        -------
        xr.DataArray
            Concatenated windows with dims
            ``('time', 'grid_index', 'window', 'forcing_feature')``.
        """
        init_steps = self.INIT_STEPS
        da_list = []

        if "analysis_time" in da_forcing.dims:
            if forecast_step is None:
                raise ValueError(
                    "forecast_step must be supplied when forcing/boundary "
                    "is in forecast mode."
                )
            # Anchor on the model init time rather than the first target,
            # so we never pick a launch that would be unavailable
            # operationally. A launch exactly at init is fine: interior
            # analysis and boundary forcing both take time to produce.
            model_init_time = cast(
                np.datetime64, state_times[init_steps - 1].values
            )
            first_target_time = cast(
                np.datetime64, state_times[init_steps].values
            )

            analysis_index = da_forcing.analysis_time.get_index("analysis_time")
            forcing_at_idx = analysis_index.get_indexer(
                [
                    _latest_usable_launch(
                        model_init_time,
                        first_target_time,
                        da_forcing.elapsed_forecast_duration.values[0],
                        forecast_step,
                        num_past_steps,
                    )
                ],
                method="pad",
            )[0]
            if forcing_at_idx < 0:
                raise ValueError(
                    "No boundary/forcing analysis time is early enough for "
                    f"the model init time ({model_init_time}) with "
                    f"{num_past_steps} past window steps."
                )
            forcing_at = da_forcing.analysis_time[forcing_at_idx]

            # `elapsed_forecast_duration` need not start at zero, so the
            # window index is measured from the first lead, not from launch.
            lead_offset = da_forcing.elapsed_forecast_duration.values[0]

            def lead_index(valid_time: np.datetime64) -> int:
                """Position of the latest lead at or before ``valid_time``."""
                return int(
                    np.floor(
                        (valid_time - forcing_at.values - lead_offset)
                        / forecast_step
                    )
                )

            for step_idx in range(len(state_times) - init_steps):
                target_time = cast(
                    np.datetime64, state_times[init_steps + step_idx].values
                )
                lead = lead_index(target_time)
                window_start = lead - num_past_steps
                window_end = lead + num_future_steps + 1
                _check_window_bounds(
                    window_start,
                    window_end,
                    da_forcing.sizes["elapsed_forecast_duration"],
                    target_time,
                    num_past_steps,
                    num_future_steps,
                    "elapsed_forecast_duration",
                )

                da_sliced = da_forcing.isel(
                    analysis_time=int(forcing_at_idx),
                    elapsed_forecast_duration=slice(
                        int(window_start), int(window_end)
                    ),
                ).rename({"elapsed_forecast_duration": "window"})
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )
                da_sliced = da_sliced.expand_dims(dim={"time": [target_time]})
                da_list.append(da_sliced)
        else:
            forcing_time_index = da_forcing.time.get_index("time")
            for step_idx in range(init_steps, len(state_times)):
                state_time = cast(np.datetime64, state_times[step_idx].values)
                forcing_time_idx = forcing_time_index.get_indexer(
                    [state_time], method="pad"
                )[0]
                if forcing_time_idx < 0:
                    raise ValueError(
                        f"No boundary/forcing time at or before {state_time}."
                    )

                window_start = forcing_time_idx - num_past_steps
                window_end = forcing_time_idx + num_future_steps + 1
                _check_window_bounds(
                    window_start,
                    window_end,
                    da_forcing.sizes["time"],
                    state_time,
                    num_past_steps,
                    num_future_steps,
                    "time",
                )

                da_window = da_forcing.isel(
                    time=slice(int(window_start), int(window_end))
                ).rename({"time": "window"})
                da_window = da_window.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )
                da_window = da_window.expand_dims(dim={"time": [state_time]})
                da_list.append(da_window)

        return xr.concat(da_list, dim="time")

    def _empty_windowed_dataarray(
        self, grid_index: xr.DataArray, target_times: xr.DataArray
    ) -> xr.DataArray:
        """Build an empty windowed forcing/boundary dataarray.

        Used when no forcing (or no boundary) is configured: the feature
        dimension has size 0 so downstream code can unpack a stable 5-tuple.

        Parameters
        ----------
        grid_index : xr.DataArray
            The ``grid_index`` coordinate to use (interior or boundary grid).
        target_times : xr.DataArray
            The ``time`` coordinate spanning the autoregressive target steps.

        Returns
        -------
        xr.DataArray
            Empty array with dims
            ``("time", "grid_index", "forcing_feature")`` and a zero-length
            feature dimension.
        """
        return xr.DataArray(
            data=np.empty((self.ar_steps, grid_index.size, 0)),
            dims=("time", "grid_index", "forcing_feature"),
            coords={
                "time": target_times,
                "grid_index": grid_index,
                "forcing_feature": [],
            },
        )

    def _build_item_dataarrays(
        self, idx: int
    ) -> tuple[
        xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
    ]:
        """
        Create the dataarrays for the initial states, target states, forcing
        and boundary data for the sample at index `idx`.

        Parameters
        ----------
        idx : int
            The index of the sample to create the dataarrays for.

        Returns
        -------
        da_init_states : xr.DataArray
            The dataarray for the initial states.
        da_target_states : xr.DataArray
            The dataarray for the target states.
        da_forcing_windowed : xr.DataArray
            The dataarray for the forcing data, windowed for the sample.
        da_boundary_windowed : xr.DataArray
            The dataarray for the boundary forcing data, windowed for the
            sample.
        da_target_times : xr.DataArray
            The dataarray for the target times.
        """
        # Handle indexing over state ensemble members. If forcing data also
        # has an ensemble dimension, we select the same member below.
        sample_idx = idx
        i_ensemble = 0
        assert self.da_state is not None

        if self.datastore.is_ensemble:
            n_ensemble_members = self.da_state.ensemble_member.size
            if not self.load_single_member:
                sample_idx, i_ensemble = divmod(idx, n_ensemble_members)
            da_state = self.da_state.isel(ensemble_member=i_ensemble)
        else:
            da_state = self.da_state

        if self.da_forcing is not None:
            if self.datastore.has_ensemble_forcing:
                da_forcing = self.da_forcing.isel(ensemble_member=i_ensemble)
            else:
                da_forcing = self.da_forcing
        else:
            da_forcing = None

        # Forcing shares the state's forecast datastore, whose analysis_time
        # series can repeat (npyfilesmeps), so it is windowed by integer
        # index; boundary is a separate datastore, windowed by time.
        da_state = self._slice_state_time(
            da_state=da_state, idx=sample_idx, n_steps=self.ar_steps
        )
        state_times = da_state["time"]

        if da_forcing is not None:
            if self.datastore.is_forecast:
                da_forcing_windowed = self._window_same_forecast_by_idx(
                    da_forcing=da_forcing,
                    idx=sample_idx,
                    state_times=state_times,
                    num_past_steps=self.num_past_forcing_steps,
                    num_future_steps=self.num_future_forcing_steps,
                )
            else:
                da_forcing_windowed = self._window_forcing_in_time(
                    da_forcing=da_forcing,
                    state_times=state_times,
                    num_past_steps=self.num_past_forcing_steps,
                    num_future_steps=self.num_future_forcing_steps,
                    forecast_step=None,
                )

        if self.da_boundary_forcing is not None:
            da_boundary_windowed = self._window_forcing_in_time(
                da_forcing=self.da_boundary_forcing,
                state_times=state_times,
                num_past_steps=self.num_past_boundary_steps,
                num_future_steps=self.num_future_boundary_steps,
                forecast_step=self._forecast_step_boundary,
            )
        else:
            da_boundary_windowed = None

        # load the data into memory
        da_state.load()
        if da_forcing is not None:
            da_forcing_windowed.load()
        if da_boundary_windowed is not None:
            da_boundary_windowed.load()

        da_init_states = da_state.isel(time=slice(0, 2))
        da_target_states = da_state.isel(time=slice(2, None))
        da_target_times = da_target_states.time

        if da_forcing is not None:
            # stack the `forcing_feature` and `window_sample` dimensions into a
            # single `forcing_feature` dimension
            da_forcing_windowed = da_forcing_windowed.stack(
                forcing_feature_windowed=("forcing_feature", "window")
            )
        else:
            da_forcing_windowed = self._empty_windowed_dataarray(
                da_state.grid_index, da_target_times
            )

        if da_boundary_windowed is not None:
            da_boundary_windowed = da_boundary_windowed.stack(
                forcing_feature_windowed=("forcing_feature", "window")
            )
        else:
            # Feature dim is empty, so the interior grid_index will do.
            da_boundary_windowed = self._empty_windowed_dataarray(
                da_state.grid_index, da_target_times
            )

        return (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_target_times,
        )

    def __getitem__(  # ty: ignore[invalid-method-override]
        self, idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Return a single training sample, which consists of the initial states,
        target states, forcing, boundary and batch times.

        The returned data is unstandardized; normalization is applied on-device
        in `ForecasterModule.on_after_batch_transfer`.

        Parameters
        ----------
        idx : int
            The index of the sample to return, this will refer to the time of
            the initial state. Negative indices follow Python sequence
            convention. Out-of-range indices raise ``IndexError``.

        Returns
        -------
        init_states : torch.Tensor
            Initial states, shape ``(2, num_grid_nodes, num_state_vars)``.
        target_states : torch.Tensor
            Target states, shape
            ``(pred_steps, num_grid_nodes, num_state_vars)``.
        forcing : torch.Tensor
            Windowed forcing, shape
            ``(pred_steps, num_grid_nodes, num_windowed_forcing_vars)`` where
            ``num_windowed_forcing_vars = num_forcing_vars``
            ``* (num_past_forcing_steps + num_future_forcing_steps + 1)``.
        boundary : torch.Tensor
            Windowed boundary forcing, shape
            ``(pred_steps, num_boundary_grid_nodes,``
            ``num_windowed_boundary_vars)``.
        target_times : torch.Tensor
            Times of the target steps, shape ``(pred_steps,)``.

        """
        n_samples = len(self)
        if idx < 0:
            idx += n_samples
        if not 0 <= idx < n_samples:
            raise IndexError(
                f"index {idx} out of range for WeatherDataset of length "
                f"{n_samples}"
            )

        (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_target_times,
        ) = self._build_item_dataarrays(idx=idx)

        tensor_dtype = torch.float32

        init_states = torch.tensor(da_init_states.values, dtype=tensor_dtype)
        target_states = torch.tensor(
            da_target_states.values, dtype=tensor_dtype
        )

        target_times = torch.tensor(
            da_target_times.astype("datetime64[ns]").astype("int64").values,
            dtype=torch.int64,
        )

        forcing = torch.tensor(da_forcing_windowed.values, dtype=tensor_dtype)
        boundary = torch.tensor(da_boundary_windowed.values, dtype=tensor_dtype)

        # init_states: (2, num_grid_nodes, num_state_vars)
        # target_states: (pred_steps, num_grid_nodes, num_state_vars)
        # forcing: (pred_steps, num_grid_nodes, num_windowed_forcing_vars)
        # boundary: (pred_steps, num_boundary_grid_nodes,
        #            num_windowed_boundary_vars)
        # target_times: (pred_steps,)

        return init_states, target_states, forcing, boundary, target_times

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
    ]:
        """
        Convenience method to iterate over the dataset.

        This isn't used by pytorch DataLoader which itself implements an
        iterator that uses Dataset.__getitem__ and Dataset.__len__.

        """
        for i in range(len(self)):
            yield self[i]

    def create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        category: str,
    ) -> xr.DataArray:
        """
        Construct a xarray.DataArray from a `pytorch.Tensor` with coordinates
        for `grid_index`, `time` and `{category}_feature` matching the shape
        and number of times provided and add the x/y coordinates from the
        datastore.

        The number if times provided is expected to match the shape of the
        tensor. For a 2D tensor, the dimensions are assumed to be (grid_index,
        {category}_feature) and only a single time should be provided. For a 3D
        tensor, the dimensions are assumed to be (time, grid_index,
        {category}_feature) and a list of times should be provided.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to construct the DataArray from, this assumed to have
            the same dimension ordering as returned by the __getitem__ method
            (i.e. time, grid_index, {category}_feature). The tensor will be
            copied to the CPU before constructing the DataArray.
        time : datetime.datetime or list[datetime.datetime]
            The time or times of the tensor.
        category : str
            The category of the tensor, either "state", "forcing" or "static".

        Returns
        -------
        da : xr.DataArray
            The constructed DataArray.
        """

        def _is_listlike(obj: object) -> bool:
            """Return ``True`` for list/tuple/ndarray-like containers."""
            return hasattr(obj, "__iter__") and not isinstance(obj, str)

        add_time_as_dim = False
        if len(tensor.shape) == 2:
            dims = ["grid_index", f"{category}_feature"]
            if _is_listlike(time):
                raise ValueError(
                    "Expected a single time for a 2D tensor with assumed "
                    "dimensions (grid_index, {category}_feature), but got "
                    f"{len(time)} times"  # ty: ignore[invalid-argument-type]
                )
        elif len(tensor.shape) == 3:
            add_time_as_dim = True
            dims = ["time", "grid_index", f"{category}_feature"]
            if not _is_listlike(time):
                raise ValueError(
                    "Expected a list of times for a 3D tensor with assumed "
                    "dimensions (time, grid_index, {category}_feature), but "
                    "got a single time"
                )
        else:
            raise ValueError(
                "Expected tensor to have 2 or 3 dimensions, but got "
                f"{len(tensor.shape)}"
            )

        da_datastore_state = getattr(self, f"da_{category}")
        da_grid_index = da_datastore_state.grid_index
        da_state_feature = da_datastore_state.state_feature

        coords = {
            f"{category}_feature": da_state_feature,
            "grid_index": da_grid_index,
        }
        if add_time_as_dim:
            coords["time"] = time

        da = xr.DataArray(
            tensor.cpu().numpy(),
            dims=dims,
            coords=coords,
        )

        for grid_coord in ["x", "y"]:
            if (
                grid_coord in da_datastore_state.coords
                and grid_coord not in da.coords
            ):
                da.coords[grid_coord] = da_datastore_state[grid_coord]

        if not add_time_as_dim:
            da.coords["time"] = time

        return da


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        datastore: BaseDatastore,
        ar_steps_train: int = 3,
        ar_steps_eval: int = 25,
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        num_past_boundary_steps: int = 1,
        num_future_boundary_steps: int = 1,
        datastore_boundary: BaseDatastore | None = None,
        load_single_member: bool = False,
        batch_size: int = 4,
        num_workers: int = 16,
        eval_split: str = "test",
    ) -> None:
        """
        Parameters
        ----------
        datastore : BaseDatastore
            Datastore used for all splits.
        ar_steps_train : int, optional
            Number of autoregressive steps for training batches. Default ``3``.
        ar_steps_eval : int, optional
            Number of autoregressive steps for validation/test batches.
            Default ``25``.
        num_past_forcing_steps : int, optional
            Number of past forcing steps to include. Default ``1``.
        num_future_forcing_steps : int, optional
            Number of future forcing steps to include. Default ``1``.
        load_single_member : bool, optional
            If ``True``, load only a single ensemble member per sample.
            Default ``False``.
        batch_size : int, optional
            Mini-batch size for dataloaders. Default ``4``.
        num_workers : int, optional
            Number of background workers per dataloader. Default ``16``.
        eval_split : str, optional
            Dataset split to use for ``test_dataloader``. Default ``"test"``.
        """
        super().__init__()
        self._datastore = datastore
        self._datastore_boundary = datastore_boundary
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.load_single_member = load_single_member
        self.batch_size = batch_size
        self.num_workers: int = num_workers
        self.train_dataset: WeatherDataset | None = None
        self.val_dataset: WeatherDataset | None = None
        self.test_dataset: WeatherDataset | None = None
        self.multiprocessing_context: str | None = None
        self.eval_split = eval_split
        if num_workers > 0:
            # default to spawn for now, as the default on linux "fork" hangs
            # when using dask (which the npyfilesmeps datastore uses)
            self.multiprocessing_context = "spawn"

    def setup(self, stage: str | None = None) -> None:
        """
        Instantiate datasets for the requested trainer stage.

        Parameters
        ----------
        stage : str or None, optional
            Trainer stage identifier (``"fit"``/``"test"``/``None``). When
            ``None``, both the training split and the validation/test
            evaluation splits are prepared.
        """
        shared_kwargs: dict[str, Any] = {
            "num_past_forcing_steps": self.num_past_forcing_steps,
            "num_future_forcing_steps": self.num_future_forcing_steps,
            "num_past_boundary_steps": self.num_past_boundary_steps,
            "num_future_boundary_steps": self.num_future_boundary_steps,
            "datastore_boundary": self._datastore_boundary,
            "load_single_member": self.load_single_member,
        }
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                datastore=self._datastore,
                split="train",
                ar_steps=self.ar_steps_train,
                **shared_kwargs,
            )
            self.val_dataset = WeatherDataset(
                datastore=self._datastore,
                split="val",
                ar_steps=self.ar_steps_eval,
                **shared_kwargs,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                split=self.eval_split,
                ar_steps=self.ar_steps_eval,
                **shared_kwargs,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Load train dataset."""
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Load validation dataset."""
        assert self.val_dataset is not None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Load test dataset."""
        assert self.test_dataset is not None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
