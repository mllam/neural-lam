# Standard library
import datetime
import warnings
from typing import Any, Iterator, Optional, Union

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseDatastore
from neural_lam.utils import (
    check_time_overlap,
    crop_time_if_needed,
    get_time_step,
)


class WeatherDataset(torch.utils.data.Dataset):
    """Dataset class for weather data.

    This class loads and processes weather data from a given datastore,
    with optional boundary forcing from a separate boundary datastore.
    Boundary windowing is aligned to interior state times by
    nearest-neighbor lookup, so the interior and boundary datastores may
    differ in step length and either side may be analysis or forecast
    data.

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
        A separate datastore providing boundary forcing data. If None, no
        boundary forcing is used (boundary tensor will be empty).
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
        datastore_boundary: Union[BaseDatastore, None] = None,
        load_single_member: bool = False,
    ) -> None:
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

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )
        if self.da_state is None:
            raise ValueError(
                "The datastore must provide state data for the WeatherDataset."
            )

        # Load boundary forcing from the boundary datastore. Alignment to
        # interior state times is done in `_window_forcing_in_time` via
        # nearest-neighbor (pad) lookup on time coordinates, so the
        # boundary datastore can have a different step length than the
        # interior, and either side may be analysis or forecast.
        if self.datastore_boundary is not None:
            self.da_boundary_forcing = self.datastore_boundary.get_dataarray(
                category="forcing", split=self.split
            )
        else:
            self.da_boundary_forcing = None

        # Within-sample time step for the state series: this is the step
        # between consecutive state times that __getitem__ exposes, used
        # below to advance the forcing/boundary window across AR steps.
        if self.datastore.is_forecast:
            self._time_step_state = get_time_step(
                self.da_state.elapsed_forecast_duration.values
            )
        else:
            self._time_step_state = get_time_step(self.da_state.time.values)

        # Forecast lead-time step for forcing/boundary, only meaningful when
        # the corresponding datastore is in forecast mode.
        self._forecast_step_forcing = None
        if self.da_forcing is not None and self.datastore.is_forecast:
            self._forecast_step_forcing = get_time_step(
                self.da_forcing.elapsed_forecast_duration.values
            )
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

            # Validate that the boundary covers the windows we will request,
            # and if necessary crop the analysis-mode interior so that the
            # very first/last samples don't fall outside boundary coverage.
            if self.da_boundary_forcing is not None:
                self.da_state = crop_time_if_needed(
                    self.da_state,
                    self.da_boundary_forcing,
                    da1_is_forecast=self.datastore.is_forecast,
                    da2_is_forecast=datastore_boundary.is_forecast,
                    num_past_steps=self.num_past_boundary_steps,
                    num_future_steps=self.num_future_boundary_steps,
                )
                check_time_overlap(
                    self.da_state,
                    self.da_boundary_forcing,
                    da1_is_forecast=self.datastore.is_forecast,
                    da2_is_forecast=datastore_boundary.is_forecast,
                    num_past_steps=self.num_past_boundary_steps,
                    num_future_steps=self.num_future_boundary_steps,
                )

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
            raise ValueError(
                "The provided datastore only provides "
                f"{len(self.da_state.time)} total time steps, which is too few "
                "to create a single sample for the WeatherDataset "
                f"configuration used in the `{split}` split. You could try "
                "either reducing the number of autoregressive steps "
                "(`ar_steps`) and/or the forcing window size "
                "(`num_past_forcing_steps` and `num_future_forcing_steps`)"
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
        da_forcing,
        state_times,
        num_past_steps: int,
        num_future_steps: int,
        forecast_step,
    ):
        """Window forcing/boundary in time, aligned to interior state times.

        ``state_times`` is the 1D ``time`` coordinate of the already-sliced
        state sample. For each AR target step the matching forcing time is
        picked by nearest-neighbor ``pad`` lookup (smallest forcing time
        ``<=`` state time), and a window of
        ``num_past_steps + num_future_steps + 1`` consecutive forcing
        entries is taken around it.

        When ``da_forcing`` has an ``analysis_time`` dimension the same
        logic is applied to forecast forcing/boundary: an analysis time is
        chosen such that the lead times cover the requested window for
        every AR step, then windows are walked across lead times.

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
            # Choose a single analysis_time (launch) for this sample. We
            # anchor on the model init time (the last input state), not the
            # first target, so we never select a boundary forecast launched
            # after init - that forecast would be unavailable operationally.
            # A launch exactly at init is also rejected (strictly before),
            # then shifted further back if a larger num_past_steps requires
            # more lead headroom.
            model_init_time = state_times[init_steps - 1].values
            first_target_time = state_times[init_steps].values

            analysis_index = da_forcing.analysis_time.get_index("analysis_time")
            forcing_at_idx = analysis_index.get_indexer(
                [model_init_time], method="pad"
            )[0]
            if forcing_at_idx < 0:
                raise ValueError(
                    "Boundary/forcing analysis times start after the model "
                    f"init time ({model_init_time})."
                )
            forcing_at = da_forcing.analysis_time[forcing_at_idx]
            if model_init_time == forcing_at.values:
                if forcing_at_idx == 0:
                    raise ValueError(
                        "No boundary/forcing analysis time strictly before "
                        f"the model init time ({model_init_time}) is available."
                    )
                forcing_at_idx -= 1
                forcing_at = da_forcing.analysis_time[forcing_at_idx]

            lead_at_first_target = int(
                np.floor(
                    (first_target_time - forcing_at.values) / forecast_step
                )
            )
            past_analysis_offset = num_past_steps - lead_at_first_target
            if past_analysis_offset > 0:
                forcing_at_idx -= past_analysis_offset
                if forcing_at_idx < 0:
                    raise ValueError(
                        "Boundary/forcing analysis times do not extend far "
                        "enough back to cover the requested past window."
                    )
                forcing_at = da_forcing.analysis_time[forcing_at_idx]

            for step_idx in range(len(state_times) - init_steps):
                target_time = state_times[init_steps + step_idx].values
                lead = int(
                    np.floor((target_time - forcing_at.values) / forecast_step)
                )
                center_time = forcing_at.values + lead * forecast_step
                assert center_time <= target_time, (
                    "Boundary forecast valid time runs ahead of the interior "
                    f"target time ({center_time} > {target_time})."
                )
                window_start = lead - num_past_steps
                window_end = lead + num_future_steps + 1

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
                state_time = state_times[step_idx].values
                forcing_time_idx = forcing_time_index.get_indexer(
                    [state_time], method="pad"
                )[0]
                if forcing_time_idx < 0:
                    raise ValueError(
                        f"No boundary/forcing time at or before {state_time}."
                    )

                window_start = forcing_time_idx - num_past_steps
                window_end = forcing_time_idx + num_future_steps + 1

                da_window = da_forcing.isel(
                    time=slice(int(window_start), int(window_end))
                ).rename({"time": "window"})
                da_window = da_window.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )
                da_window = da_window.expand_dims(dim={"time": [state_time]})
                da_list.append(da_window)

        return xr.concat(da_list, dim="time")

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

        # Slice the state once, then window forcing and boundary against
        # the resulting state times. Forcing is windowed by integer
        # `analysis_time` index when it comes from the same forecast
        # datastore as state (the analysis_time series can have repeats
        # there, e.g. npyfilesmeps); boundary always comes from a
        # different datastore so it is windowed by time-based
        # nearest-neighbor lookup.
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
            # create an empty forcing tensor with the right shape
            da_forcing_windowed = xr.DataArray(
                data=np.empty(
                    (self.ar_steps, da_state.grid_index.size, 0),
                ),
                dims=("time", "grid_index", "forcing_feature"),
                coords={
                    "time": da_target_times,
                    "grid_index": da_state.grid_index,
                    "forcing_feature": [],
                },
            )

        if da_boundary_windowed is not None:
            da_boundary_windowed = da_boundary_windowed.stack(
                forcing_feature_windowed=("forcing_feature", "window")
            )
        else:
            # create an empty boundary tensor with the right shape
            # Use the boundary datastore's grid_index if available, otherwise
            # fall back to state grid_index (for the no-boundary case the
            # last dim is 0 anyway)
            if self.datastore_boundary is not None:
                da_boundary_ref = self.datastore_boundary.get_dataarray(
                    category="forcing", split=self.split
                )
                boundary_grid_index = (
                    da_boundary_ref.grid_index
                    if da_boundary_ref is not None
                    else da_state.grid_index
                )
            else:
                boundary_grid_index = da_state.grid_index
            da_boundary_windowed = xr.DataArray(
                data=np.empty(
                    (self.ar_steps, boundary_grid_index.size, 0),
                ),
                dims=("time", "grid_index", "forcing_feature"),
                coords={
                    "time": da_target_times,
                    "grid_index": boundary_grid_index,
                    "forcing_feature": [],
                },
            )

        return (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_target_times,
        )

    def __getitem__(
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
            Initial states, shape (2, N_grid, d_features).
        target_states : torch.Tensor
            Target states, shape (ar_steps, N_grid, d_features).
        forcing : torch.Tensor
            Windowed forcing, shape (ar_steps, N_grid, d_windowed_forcing).
        boundary : torch.Tensor
            Windowed boundary forcing, shape
            (ar_steps, N_boundary_grid, d_windowed_boundary).
        target_times : torch.Tensor
            Times of the target steps, shape (ar_steps,).

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

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # boundary: (ar_steps, N_boundary_grid, d_windowed_boundary)
        # target_times: (ar_steps,)

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
        time: Union[datetime.datetime, list[datetime.datetime]],
        category: str,
    ):
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

        def _is_listlike(obj):
            # match list, tuple, numpy array
            return hasattr(obj, "__iter__") and not isinstance(obj, str)

        add_time_as_dim = False
        if len(tensor.shape) == 2:
            dims = ["grid_index", f"{category}_feature"]
            if _is_listlike(time):
                raise ValueError(
                    "Expected a single time for a 2D tensor with assumed "
                    "dimensions (grid_index, {category}_feature), but got "
                    f"{len(time)} times"  # type: ignore
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
        datastore_boundary: Union[BaseDatastore, None] = None,
        load_single_member: bool = False,
        batch_size: int = 4,
        num_workers: int = 16,
        eval_split: str = "test",
    ) -> None:
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
        self.train_dataset: Optional[WeatherDataset] = None
        self.val_dataset: Optional[WeatherDataset] = None
        self.test_dataset: Optional[WeatherDataset] = None
        self.multiprocessing_context: Union[str, None] = None
        self.eval_split = eval_split
        if num_workers > 0:
            # default to spawn for now, as the default on linux "fork" hangs
            # when using dask (which the npyfilesmeps datastore uses)
            self.multiprocessing_context = "spawn"

    def setup(self, stage: Optional[str] = None) -> None:
        shared_kwargs: dict[str, Any] = dict(
            num_past_forcing_steps=self.num_past_forcing_steps,
            num_future_forcing_steps=self.num_future_forcing_steps,
            num_past_boundary_steps=self.num_past_boundary_steps,
            num_future_boundary_steps=self.num_future_boundary_steps,
            datastore_boundary=self._datastore_boundary,
            load_single_member=self.load_single_member,
        )
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
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
