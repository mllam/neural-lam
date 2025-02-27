# Standard library
import datetime
import warnings
from typing import Union

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

    This class loads and processes weather data from a given datastore.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore to load the data from.
    datastore_boundary : BaseDatastore
        The boundary datastore to load the data from.
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
        Number of past time steps to include in boundary input. If set to i,
        boundary from times t-i, t-i+1, ..., t-1, t (and potentially beyond,
        given num_future_forcing_steps) are included as boundary inputs at time
        t Default is 1.
    num_future_boundary_steps: int, optional
        Number of future time steps to include in boundary input. If set to j,
        boundary from times t, t+1, ..., t+j-1, t+j (and potentially times
        before t, given num_past_forcing_steps) are included as boundary inputs
        at time t. Default is 1.
    interior_subsample_step : int, optional
        The stride/step size used when sampling interior domain data points. A
        value of N means only every Nth point will be sampled in the temporal
        dimension. For example, if step_length=3 hours and
        interior_subsample_step=2, data will be sampled every 6 hours. Default
        is 1 (use every timestep).
    boundary_subsample_step : int, optional
        The stride/step size used when sampling boundary condition data points.
        A value of N means only every Nth point will be sampled in the temporal
        dimension. For example, if step_length=3 hours and
        boundary_subsample_step=2, boundary conditions will be sampled every 6
        hours. Default is 1 (use every timestep).
    standardize : bool, optional
        Whether to standardize the data. Default is True.
    dynamic_time_deltas : bool, optional
        If time-deltas of boundary time steps should be dynamically computed as
        time between interior and boundary.
    """

    # The current implementation requires at least 2 time steps for the
    # initial state (see GraphCast).
    INIT_STEPS = 2  # Number of initial state steps needed

    def __init__(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore,
        split="train",
        ar_steps=3,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        interior_subsample_step=1,
        boundary_subsample_step=1,
        standardize=True,
        dynamic_time_deltas=False,
        time_slice=None,
    ):
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.datastore = datastore
        self.datastore_boundary = datastore_boundary
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.interior_subsample_step = interior_subsample_step
        self.boundary_subsample_step = boundary_subsample_step
        self.dynamic_time_deltas = dynamic_time_deltas
        # Scale forcing steps based on subsampling
        self.effective_past_forcing_steps = (
            num_past_forcing_steps * interior_subsample_step
        )
        self.effective_future_forcing_steps = (
            num_future_forcing_steps * interior_subsample_step
        )
        self.effective_past_boundary_steps = (
            num_past_boundary_steps * boundary_subsample_step
        )
        self.effective_future_boundary_steps = (
            num_future_boundary_steps * boundary_subsample_step
        )

        # Validate subsample steps
        if (
            not isinstance(interior_subsample_step, int)
            or interior_subsample_step < 1
        ):
            raise ValueError(
                "interior_subsample_step must be a positive integer"
            )
        if (
            not isinstance(boundary_subsample_step, int)
            or boundary_subsample_step < 1
        ):
            raise ValueError(
                "boundary_subsample_step must be a positive integer"
            )

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        if self.da_state is None:
            raise ValueError(
                "A non-empty state dataarray must be provided. "
                "The datastore.get_dataarray() returned None or empty array "
                "for category='state'"
            )
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )
        # XXX For now boundary data is always considered mdp-forcing data
        if self.datastore_boundary is not None:
            self.da_boundary_forcing = self.datastore_boundary.get_dataarray(
                category="forcing", split=self.split
            )
        else:
            self.da_boundary_forcing = None

        # Do not need to subset boundary forcing
        # split arg is never used past this point
        if time_slice is not None:
            print(
                "Created WeatherDataset for time interval "
                f"{time_slice.start} - {time_slice.stop}"
            )
            # Subset state, forcing, boundary_forcing
            self.da_state = self.da_state.sel(time=time_slice)
            self.da_forcing = self.da_forcing.sel(time=time_slice)
            self.da_boundary_forcing = self.da_boundary_forcing.sel(
                time=time_slice
            )

        # check that with the provided data-arrays and ar_steps that we have a
        # non-zero amount of samples
        if self.__len__() <= 0:
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
            expected_dim_order = self.datastore.expected_dim_order(
                category=part
            )
            if da.dims != expected_dim_order:
                raise ValueError(
                    f"The dimension order of the `{part}` data ({da.dims}) "
                    f"does not match the expected dimension order "
                    f"({expected_dim_order}). Maybe you forgot to transpose "
                    "the data in `BaseDatastore.get_dataarray`?"
                )

        # handling ensemble data
        if self.datastore.is_ensemble:
            # for the now the strategy is to only include the first ensemble
            # member
            # XXX: this could be changed to include all ensemble members by
            # splitting `idx` into two parts, one for the analysis time and one
            # for the ensemble member and then increasing self.__len__ to
            # include all ensemble members
            warnings.warn(
                "only use of ensemble member 0 (the first member) is "
                "implemented for ensemble data"
            )
            i_ensemble = 0
            self.da_state = self.da_state.isel(ensemble_member=i_ensemble)

        # Check time step consistency in state data and determine time steps
        # for state, forcing and boundary forcing data
        # STATE
        if self.datastore.is_forecast:
            state_times = self.da_state.analysis_time
            self.forecast_step_state = get_time_step(
                self.da_state.elapsed_forecast_duration
            )
        else:
            state_times = self.da_state.time
        self.orig_time_step_state = get_time_step(state_times)
        self.time_step_state = (
            self.interior_subsample_step * self.orig_time_step_state
        )
        # FORCING
        if self.da_forcing is not None:
            if self.datastore.is_forecast:
                forcing_times = self.da_forcing.analysis_time
                self.forecast_step_forcing = get_time_step(
                    self.da_forcing.elapsed_forecast_duration
                )
            else:
                forcing_times = self.da_forcing.time
            self.time_step_forcing = (
                self.boundary_subsample_step
                * get_time_step(forcing_times.values)
            )
        # inform user about the original and the subsampled time step
        if self.interior_subsample_step != 1:
            print(
                f"Subsampling interior data with step size "
                f"{self.interior_subsample_step} from original time step "
                f"{self.orig_time_step_state}"
            )
        else:
            print(
                f"Using original time step {self.orig_time_step_state} for data"
            )

        # BOUNDARY FORCING
        if self.da_boundary_forcing is not None:
            if self.datastore_boundary.is_forecast:
                boundary_times = self.da_boundary_forcing.analysis_time
                self.forecast_step_boundary = get_time_step(
                    self.da_boundary_forcing.elapsed_forecast_duration
                )
            else:
                boundary_times = self.da_boundary_forcing.time
            self.time_step_boundary = get_time_step(boundary_times.values)

            if self.boundary_subsample_step != 1:
                print(
                    f"Subsampling boundary data with step size "
                    f"{self.boundary_subsample_step} from original time step "
                    f"{self.time_step_boundary}"
                )
            else:
                print(
                    f"Using original time step {self.time_step_boundary} for "
                    "boundary data"
                )

        # Forcing data is part of the same datastore as state data. During
        # creation, the time dimension of the forcing data is matched to the
        # state data.
        # Boundary data is part of a separate datastore The boundary data is
        # allowed to have a different time_step Checks that the boundary data
        # covers the required time range is required.
        # Crop interior data if boundary coverage is insufficient
        if self.da_boundary_forcing is not None:
            self.da_state = crop_time_if_needed(
                self.da_state,
                self.da_boundary_forcing,
                da1_is_forecast=self.datastore.is_forecast,
                da2_is_forecast=self.datastore_boundary.is_forecast,
                num_past_steps=self.num_past_boundary_steps,
                num_future_steps=self.num_future_boundary_steps,
            )

        # Now do final overlap check and possibly raise errors if still invalid
        if self.da_boundary_forcing is not None:
            check_time_overlap(
                self.da_state,
                self.da_boundary_forcing,
                da1_is_forecast=self.datastore.is_forecast,
                da2_is_forecast=self.datastore_boundary.is_forecast,
                num_past_steps=self.num_past_boundary_steps,
                num_future_steps=self.num_future_boundary_steps,
            )

        # check that also after cropping we have a non-zero amount of samples
        if self.__len__() <= 0:
            raise ValueError(
                "The provided datastore (after cropping) only provides "
                f"{len(self.da_state.time)} total time steps, which is too few "
                "to create a single sample for the WeatherDataset "
                f"configuration used in the `{split}` split. You could try "
                "either reducing the number of autoregressive steps "
                "(`ar_steps`) and/or the forcing window size "
                "(`num_past_forcing_steps` and `num_future_forcing_steps`)"
            )

        # Set up for standardization
        # TODO: This will become part of ar_model.py soon!
        self.standardize = standardize
        if standardize:
            self.ds_state_stats = self.datastore.get_standardization_dataarray(
                category="state"
            )

            self.da_state_mean = self.ds_state_stats.state_mean
            self.da_state_std = self.ds_state_stats.state_std

            if self.da_forcing is not None:
                self.ds_forcing_stats = (
                    self.datastore.get_standardization_dataarray(
                        category="forcing"
                    )
                )
                self.da_forcing_mean = self.ds_forcing_stats.forcing_mean
                self.da_forcing_std = self.ds_forcing_stats.forcing_std

            # XXX: Again, the boundary data is considered forcing data for now
            if self.da_boundary_forcing is not None:
                self.ds_boundary_stats = (
                    self.datastore_boundary.get_standardization_dataarray(
                        category="forcing"
                    )
                )
                self.da_boundary_mean = self.ds_boundary_stats.forcing_mean
                self.da_boundary_std = self.ds_boundary_stats.forcing_std

    def __len__(self):
        if self.datastore.is_ensemble:
            warnings.warn(
                "only using first ensemble member, so dataset size is "
                " effectively reduced by the number of ensemble members "
                f"({self.datastore.num_ensemble_members})",
                UserWarning,
            )

        if self.datastore.is_forecast:
            # for now we simply create a single sample for each analysis time
            # and then take the first (2 + ar_steps) forecast times. In
            # addition we only use the first ensemble member (if ensemble data
            # has been provided).
            # This means that for each analysis time we get a single sample
            # check that there are enough forecast steps available to create
            # samples given the number of autoregressive steps requested
            required_steps = self.INIT_STEPS + self.ar_steps
            required_span = (required_steps - 1) * self.interior_subsample_step

            # Calculate available forecast steps
            n_forecast_steps = len(self.da_state.elapsed_forecast_duration)

            if n_forecast_steps < required_span:
                raise ValueError(
                    f"Not enough forecast steps ({n_forecast_steps}) for "
                    f"required span of {required_span} steps with "
                    f"subsample_step={self.interior_subsample_step}"
                )

            return self.da_state.analysis_time.size
        else:
            # Calculate the number of samples in the dataset as:
            # total_samples = total_timesteps - required_time_span -
            # required_past_steps - effective_future_forcing_steps
            # Where:
            # - total_timesteps: total number of timesteps in the state data
            # - required_time_span: number of continuous timesteps needed for
            #   initial state + autoregressive steps, accounting for subsampling
            # - required_past_steps: additional past timesteps needed for
            #   forcing data beyond initial state
            # - effective_future_forcing_steps: number of future timesteps
            #   needed for forcing data with subsampling
            required_continuous_steps = self.INIT_STEPS + self.ar_steps
            required_time_span = (
                required_continuous_steps * self.interior_subsample_step
            )
            required_past_steps = max(
                0,
                self.effective_past_forcing_steps
                - self.INIT_STEPS * self.interior_subsample_step,
            )

            return (
                len(self.da_state.time)
                - required_time_span
                - required_past_steps
                - self.effective_future_forcing_steps
            )

    def _slice_time(
        self,
        da_state,
        idx,
        n_steps: int,
        da_forcing=None,
        num_past_steps=None,
        num_future_steps=None,
        is_boundary=False,
    ):
        """
        Produce time slices of the given dataarrays `da_state` (state) and
        `da_forcing`. For the state data, slicing is done based on `idx`. For
        the forcing/boundary data, nearest neighbor matching is performed based
        on the state times (assuming constant timestep size). Additionally, the
        time deltas between the matched forcing/boundary times and state times
        (in multiples of state time steps) is added to the forcing dataarray.
        This will be used as an additional input feature in the model (as
        temporal embedding).

        Parameters
        ----------
        da_state : xr.DataArray
            The state dataarray to slice.
        idx : int
            The index of the time step to start the sample from in the state
            data.
        n_steps : int
            The number of time steps to include in the sample.
        da_forcing : xr.DataArray
            The forcing/boundary dataarray to slice.
        num_past_steps : int, optional
            The number of past time steps to include in the forcing/boundary
            data. Default is `None`.
        num_future_steps : int, optional
            The number of future time steps to include in the forcing/boundary
            data. Default is `None`.
        is_boundary : bool, optional
            Whether the data is boundary data. Default is `False`.

        Returns
        -------
        da_state_sliced : xr.DataArray
            The sliced state dataarray with dims ('time', 'grid_index',
            'state_feature').
        da_forcing_matched : xr.DataArray
            The sliced state dataarray with dims ('time', 'grid_index',
            'forcing/boundary_feature_windowed').
            If no forcing/boundary data is provided, this will be `None`.
        """
        init_steps = self.INIT_STEPS
        subsample_step = (
            self.boundary_subsample_step
            if is_boundary
            else self.interior_subsample_step
        )
        # slice the dataarray to include the required number of time steps
        if self.datastore.is_forecast:
            # this implies that the data will have both `analysis_time` and
            # `elapsed_forecast_duration` dimensions for forecasts. We for now
            # simply select a analysis time and the first `n_steps` forecast
            # times (given no offset). Note that this means that we get one
            # sample per forecast, always starting at forecast time 2.

            # Calculate base offset and indices with subsampling
            offset = (
                max(0, num_past_steps - init_steps) if num_past_steps else 0
            )

            # Calculate initial and target indices
            init_indices = [
                offset + i * subsample_step for i in range(init_steps)
            ]
            target_indices = [
                offset + (init_steps + i) * subsample_step
                for i in range(n_steps)
            ]
            all_indices = init_indices + target_indices

            da_state_sliced = da_state.isel(
                analysis_time=idx,
                elapsed_forecast_duration=all_indices,
            )
            da_state_sliced["time"] = (
                da_state_sliced.analysis_time
                + da_state_sliced.elapsed_forecast_duration
            )
            da_state_sliced = da_state_sliced.swap_dims(
                {"elapsed_forecast_duration": "time"}
            )

        else:
            # Analysis data slicing, already correctly modified
            start_idx = idx + (
                max(0, num_past_steps - init_steps) if num_past_steps else 0
            )
            all_indices = [
                start_idx + i * subsample_step
                for i in range(init_steps + n_steps)
            ]
            da_state_sliced = da_state.isel(time=all_indices)

        if da_forcing is None:
            return da_state_sliced, None

        # Get the state times and its temporal resolution for matching with
        # forcing data.
        state_times = da_state_sliced["time"]
        da_list = []
        # Here we cannot check 'self.datastore.is_forecast' directly because we
        # might be dealing with a datastore_boundary
        if "analysis_time" in da_forcing.dims:
            # For forecast data with analysis_time and elapsed_forecast_duration
            # Select the closest analysis_time in the past (strictly) in the
            # boundary data
            model_init_time = state_times[init_steps - 1].values
            # Find first index before
            forcing_analysis_time_idx = da_forcing.analysis_time.get_index(
                "analysis_time"
            ).get_indexer([model_init_time], method="pad")[0]
            forcing_analysis_time = da_forcing.analysis_time[
                forcing_analysis_time_idx
            ]
            if model_init_time == forcing_analysis_time:
                # Can not use boundary forcing initialized at same time,
                # take one before
                forcing_analysis_time_idx = forcing_analysis_time_idx - 1
                forcing_analysis_time = da_forcing.analysis_time[
                    forcing_analysis_time_idx
                ]

            # With current forcing_analysis_time_idx, how much space is there
            # for including previous time steps
            cur_prev_steps_in_forcing = (
                np.floor(
                    (model_init_time - forcing_analysis_time)
                    / self.forecast_step_boundary
                )
            ).astype(int)

            # There will always be space for 1 past_forcing step,
            # but more might require using an earlier forecast
            # We will gain 1 to possible past windowing by each index we offset
            past_analysis_offset = num_past_steps - cur_prev_steps_in_forcing
            if past_analysis_offset > 0:
                forcing_analysis_time_idx = (
                    forcing_analysis_time_idx - past_analysis_offset
                )
                forcing_analysis_time = da_forcing.analysis_time[
                    forcing_analysis_time_idx
                ]

            # Index of elapsed_forecast_duration that matches model_init_time
            forcing_lead_i_init = (
                np.floor(
                    (model_init_time - forcing_analysis_time)
                    / self.forecast_step_boundary
                )
            ).astype(int)

            forcing_first_valid_time = (
                forcing_analysis_time
                + da_forcing.elapsed_forecast_duration[forcing_lead_i_init]
            )
            # How far "behind" the init time do forcing times start from
            start_time_delay_fraction = (
                model_init_time - forcing_first_valid_time
            ) / self.forecast_step_boundary

            # Adjust window indices for subsampled steps
            for step_idx in range(len(state_times) - init_steps):
                # Figure out how many steps to offset window,
                # if time steps don't align this is not step_idx steps

                step_window_offset = np.floor(
                    start_time_delay_fraction
                    + step_idx
                    * self.time_step_state
                    / self.forecast_step_boundary
                ).astype(int)
                window_start = (
                    forcing_lead_i_init
                    + step_window_offset
                    - num_past_steps * subsample_step
                ).values
                window_end = (
                    forcing_lead_i_init
                    + step_window_offset
                    + (num_future_steps + 1) * subsample_step
                ).values

                # Time at which boundary forcing is valid
                current_time = (
                    forcing_analysis_time
                    + da_forcing.elapsed_forecast_duration[
                        forcing_lead_i_init + step_window_offset
                    ]
                )

                # Check that boundary and state times align
                # They do not have to be the same, but boundary time should
                # not be less than a boundary time step before state time
                cur_state_time = state_times[1 + step_idx]

                assert current_time <= cur_state_time, (
                    "Mismatch in boundary (forecast) and interior state times:"
                    f"boundary forcing at time {current_time.values}"
                    f"matched to state time {cur_state_time.values}"
                )
                boundary_state_time_diff = cur_state_time - current_time
                assert (current_time <= cur_state_time) and (
                    boundary_state_time_diff < self.forecast_step_boundary
                ), (
                    "Mismatch in boundary (forecast) and interior state times:"
                    f"boundary forcing at time {current_time.values}"
                    f"matched to state time {cur_state_time.values}"
                )

                da_sliced = da_forcing.isel(
                    analysis_time=forcing_analysis_time_idx,
                    elapsed_forecast_duration=slice(
                        window_start, window_end, subsample_step
                    ),
                )
                da_sliced = da_sliced.rename(
                    {"elapsed_forecast_duration": "window"}
                )

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )
                # Calculate window time deltas for forecast data
                if self.dynamic_time_deltas:
                    # Deltas compared to interior state time
                    window_comp_time = (
                        model_init_time + step_idx * self.time_step_state
                    )
                else:
                    window_comp_time = (
                        forcing_analysis_time
                        + da_forcing.elapsed_forecast_duration[
                            forcing_lead_i_init
                        ]
                    )

                window_times = (
                    forcing_analysis_time
                    + da_forcing.elapsed_forecast_duration[
                        window_start:window_end:subsample_step
                    ]
                )
                window_time_deltas = (window_times - window_comp_time).values

                # Assign window time delta coordinate
                da_sliced["window_time_deltas"] = ("window", window_time_deltas)

                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

        else:
            for idx_time in range(init_steps, len(state_times)):
                state_time = state_times[idx_time].values

                # Select the closest time in the past from forcing data using
                # sel with method="pad"
                forcing_time_idx = da_forcing.time.get_index(
                    "time"
                ).get_indexer([state_time], method="pad")[0]

                window_start = (
                    forcing_time_idx - num_past_steps * subsample_step
                )
                window_end = (
                    forcing_time_idx + (num_future_steps + 1) * subsample_step
                )

                da_window = da_forcing.isel(
                    time=slice(window_start, window_end, subsample_step)
                )

                # Rename the time dimension to window for consistency
                da_window = da_window.rename({"time": "window"})

                # Assign the 'window' coordinate to be relative positions
                da_window = da_window.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )

                # Calculate window time deltas for analysis data
                if self.dynamic_time_deltas:
                    # Deltas compared to interior state time
                    window_comp_time = state_time
                else:
                    window_comp_time = da_forcing.time[forcing_time_idx].values

                window_time_deltas = (
                    da_forcing.time[
                        window_start:window_end:subsample_step
                    ].values
                    - window_comp_time
                )
                da_window["window_time_deltas"] = ("window", window_time_deltas)

                da_window = da_window.expand_dims(dim={"time": [state_time]})

                da_list.append(da_window)

        da_forcing_matched = xr.concat(da_list, dim="time")

        return da_state_sliced, da_forcing_matched

    def _process_windowed_data(
        self, da_windowed, da_state, da_target_times, add_time_deltas=True
    ):
        """Helper function to process windowed data. This function stacks the
        'forcing_feature' and 'window' dimensions and adds the time step
        deltas to the existing features.

        Parameters
        ----------
        da_windowed : xr.DataArray
            The windowed data to process. Can be `None` if no data is provided.
        da_state : xr.DataArray
            The state dataarray.
        da_target_times : xr.DataArray
            The target times.
        add_time_deltas : bool
            If time deltas to each window position should be concatenated
            as features

        Returns
        -------
        da_windowed : xr.DataArray
            The processed windowed data. If `da_windowed` is `None`, an empty
            DataArray with the correct dimensions and coordinates is returned.

        """
        stacked_dim = "forcing_feature_windowed"
        if da_windowed is not None:
            window_size = da_windowed.window.size
            # Stack the 'feature' and 'window' dimensions and add the
            # time deltas to the existing features
            da_windowed = da_windowed.stack(
                {stacked_dim: ("forcing_feature", "window")}
            )
            if add_time_deltas:
                # Add the time deltas a new feature to the windowed
                # data, as a multiple of the state time step
                time_deltas = (
                    da_windowed["window_time_deltas"].isel(
                        forcing_feature_windowed=slice(0, window_size)
                    )
                    / self.time_step_state
                )
                # All data variables share the same time deltas
                da_windowed = xr.concat(
                    [da_windowed, time_deltas],
                    dim="forcing_feature_windowed",
                )
        else:
            # Create empty DataArray with the correct dimensions and coordinates
            da_windowed = xr.DataArray(
                data=np.empty((self.ar_steps, da_state.grid_index.size, 0)),
                dims=("time", "grid_index", f"{stacked_dim}"),
                coords={
                    "time": da_target_times,
                    "grid_index": da_state.grid_index,
                    f"{stacked_dim}": [],
                },
            )
        return da_windowed

    def _build_item_dataarrays(self, idx):
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
            The dataarray for the boundary data, windowed for the sample.
            Boundary data is always considered forcing data.
        da_target_times : xr.DataArray
            The dataarray for the target times.
        """
        da_state = self.da_state
        if self.da_forcing is not None:
            if "ensemble_member" in self.da_forcing.dims:
                raise NotImplementedError(
                    "Ensemble member not yet supported for forcing data"
                )
            da_forcing = self.da_forcing
        else:
            da_forcing = None

        if self.da_boundary_forcing is not None:
            da_boundary = self.da_boundary_forcing
        else:
            da_boundary = None

        # This function will return a slice of the state data and the forcing
        # and boundary data (if provided) for one sample (idx).
        # If da_forcing is None, the function will return None for
        # da_forcing_windowed.
        if da_boundary is not None:
            _, da_boundary_windowed = self._slice_time(
                da_state=da_state,
                idx=idx,
                n_steps=self.ar_steps,
                da_forcing=da_boundary,
                num_future_steps=self.num_future_boundary_steps,
                num_past_steps=self.num_past_boundary_steps,
                is_boundary=True,
            )
        else:
            da_boundary_windowed = None
            # XXX: Currently, the order of the `slice_time` calls is important
            # as `da_state` is modified in the second call. This should be
            # refactored to be more robust.
        da_state, da_forcing_windowed = self._slice_time(
            da_state=da_state,
            idx=idx,
            n_steps=self.ar_steps,
            da_forcing=da_forcing,
            num_future_steps=self.num_future_forcing_steps,
            num_past_steps=self.num_past_forcing_steps,
        )

        # load the data into memory
        da_state.load()
        if da_forcing is not None:
            da_forcing_windowed.load()
        if da_boundary is not None:
            da_boundary_windowed.load()

        da_init_states = da_state.isel(time=slice(0, 2))
        da_target_states = da_state.isel(time=slice(2, None))
        da_target_times = da_target_states.time

        if self.standardize:
            da_init_states = (
                da_init_states - self.da_state_mean
            ) / self.da_state_std
            da_target_states = (
                da_target_states - self.da_state_mean
            ) / self.da_state_std

            if da_forcing is not None:
                # XXX: Here we implicitly assume that the last dimension of the
                # forcing data is the forcing feature dimension. To standardize
                # on `.device` we need a different implementation. (e.g. a
                # tensor with repeated means and stds for each "windowed" time.)
                da_forcing_windowed = (
                    da_forcing_windowed - self.da_forcing_mean
                ) / self.da_forcing_std

            if da_boundary is not None:
                da_boundary_windowed = (
                    da_boundary_windowed - self.da_boundary_mean
                ) / self.da_boundary_std

        # This function handles the stacking of the forcing and boundary data
        # and adds the time deltas. It can handle `None` inputs for the forcing
        # and boundary data (and simlpy return an empty DataArray in that case).
        # We don't need time delta features for interior forcing, as these
        # deltas are always the same.
        da_forcing_windowed = self._process_windowed_data(
            da_forcing_windowed,
            da_state,
            da_target_times,
            add_time_deltas=False,
        )
        da_boundary_windowed = self._process_windowed_data(
            da_boundary_windowed,
            da_state,
            da_target_times,
            add_time_deltas=True,
        )

        return (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
            da_boundary_windowed,
            da_target_times,
        )

    def __getitem__(self, idx):
        """
        Return a single training sample, which consists of the initial states,
        target states, forcing and batch times.

        The implementation currently uses xarray.DataArray objects for the
        standardization (scaling to mean 0.0 and standard deviation of 1.0) so
        that we can make us of xarray's broadcasting capabilities. This makes
        it possible to standardization with both global means, but also for
        example where a grid-point mean has been computed. This code will have
        to be replace if standardization is to be done on the GPU to handle
        different shapes of the standardization.

        Parameters
        ----------
        idx : int
            The index of the sample to return, this will refer to the time of
            the initial state.

        Returns
        -------
        init_states : TrainingSample
            A training sample object containing the initial states, target
            states, forcing and batch times. The batch times are the times of
            the target steps.

        """
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
        # boundary: (ar_steps, N_grid, d_windowed_boundary)
        # target_times: (ar_steps,)

        # Assert that the boundary data is an empty tensor if the corresponding
        # datastore_boundary is `None`
        if self.datastore_boundary is None:
            assert boundary.numel() == 0

        return init_states, target_states, forcing, boundary, target_times

    def __iter__(self):
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
                    f"{len(time)} times"
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

        tensor = tensor.detach().cpu().numpy()
        da = xr.DataArray(
            tensor,
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


class EvalSubsetWrapper(torch.utils.data.Dataset):
    """
    A PyTorch Dataset wrapper that selects only samples with specific
    initialization times.

    Args:
        dataset: The base dataset to wrap
        eval_init_times: initialization times to include
    """

    def __init__(self, dataset, eval_init_times):
        self.dataset = dataset
        self.eval_init_times = eval_init_times

        # TODO generalize class beyond 00/12 UTC
        assert self.eval_init_times == [
            0,
            12,
        ], "Only eval_init_times 00, 12 implemented"
        valid_init_diff = 12

        # Figure out which indices to use
        first_batch = dataset[0]
        first_init_time = self.get_utc_init_of_batch(first_batch)

        # Note that we have to consider orig_time_step-state, as that is the
        # time difference between init times in self.dataset
        time_step_state_hour = self.dataset.orig_time_step_state.astype(
            "timedelta64[h]"
        ).astype(int)
        assert (
            valid_init_diff % time_step_state_hour == 0
        ), "Invalid time step for eval_init_times"

        # Find how many indices to skip in each step
        self.valid_idx_interval = valid_init_diff // time_step_state_hour

        # Find first valid index
        first_valid_idx = None
        for idx, step_offset in enumerate(range(0, 24, time_step_state_hour)):
            # Init time in h UTC
            potential_init_time = (first_init_time + step_offset) % 24
            if potential_init_time in eval_init_times:
                first_valid_idx = idx
                break

        assert first_valid_idx is not None, "Found no valid init time"
        self.first_valid_idx = first_valid_idx

    def get_utc_init_of_batch(self, batch):
        """Get init time for batch in UTC, as int"""
        target_times_np = batch[-1].numpy().astype("datetime64[ns]")
        init_time = target_times_np[0] - self.dataset.orig_time_step_state
        init_time_hour = init_time.astype("datetime64[h]").astype(int) % 24
        return init_time_hour

    def __len__(self) -> int:
        """
        Returns the length of the filtered dataset.
        """
        orig_len = len(self.dataset)
        return np.ceil(
            (orig_len - self.first_valid_idx) / self.valid_idx_interval
        ).astype(int)

    def __getitem__(self, idx: int):
        """
        Get an item from the filtered dataset.

        Args:
            idx: Index into the filtered dataset

        Returns:
            The dataset item corresponding to the filtered index
        """
        new_idx = self.first_valid_idx + idx * self.valid_idx_interval
        batch = self.dataset[new_idx]

        # Assert that we only do forecasts from 00 and 12 UTC
        analysis_time_hour = self.get_utc_init_of_batch(batch)
        assert analysis_time_hour in self.eval_init_times, (
            "Tried to evaluate on sample with "
            f"analysis time {analysis_time_hour}"
        )

        return batch


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        datastore: BaseDatastore,
        datastore_boundary: BaseDatastore,
        ar_steps_train=3,
        ar_steps_eval=25,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        interior_subsample_step=1,
        boundary_subsample_step=1,
        batch_size=4,
        num_workers=16,
        eval_split="test",
        eval_init_times=[],
        dynamic_time_deltas=False,
        excluded_intervals=None,
    ):
        super().__init__()
        self._datastore = datastore
        self._datastore_boundary = datastore_boundary
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.interior_subsample_step = interior_subsample_step
        self.boundary_subsample_step = boundary_subsample_step
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.eval_split = eval_split
        self.eval_init_times = eval_init_times
        self.dynamic_time_deltas = dynamic_time_deltas

        if num_workers > 0:
            # BUG: There also seem to be issues with "spawn" and `gloo`, to be
            # investigated. Defaults to spawn for now, as the default on linux
            # "fork" hangs when using dask (which the npyfilesmeps datastore
            # uses)
            self.multiprocessing_context = "spawn"
        else:
            self.multiprocessing_context = None

        if excluded_intervals:
            # Convert to np.datetime64
            self.excluded_intervals = [
                (np.datetime64(start_time), np.datetime64(end_time))
                for (start_time, end_time) in excluded_intervals
            ]
            for time_interval in self.excluded_intervals:
                assert time_interval[0] <= time_interval[1], (
                    "Can not exclude a time interval from "
                    f"{time_interval[0]} to {time_interval[1]}"
                )
        else:
            self.excluded_intervals = []

    def make_training_dataset(self, time_slice):
        return WeatherDataset(
            datastore=self._datastore,
            datastore_boundary=self._datastore_boundary,
            split="train",
            ar_steps=self.ar_steps_train,
            standardize=self.standardize,
            num_past_forcing_steps=self.num_past_forcing_steps,
            num_future_forcing_steps=self.num_future_forcing_steps,
            num_past_boundary_steps=self.num_past_boundary_steps,
            num_future_boundary_steps=self.num_future_boundary_steps,
            interior_subsample_step=self.interior_subsample_step,
            boundary_subsample_step=self.boundary_subsample_step,
            dynamic_time_deltas=self.dynamic_time_deltas,
            time_slice=time_slice,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.excluded_intervals:
                # Swiss-cheese strategy
                # Figure out which time-intervals to include
                full_time_interval = self._datastore.get_dataarray(
                    category="state", split="train"
                ).time

                # Iterate over times and exclude intervals
                step_length = self._datastore.step_length
                ds_intervals = []
                # Start time of current interval
                interval_start_time = full_time_interval[0].to_numpy()
                for exc_start, exc_end in self.excluded_intervals:
                    # Add interval up to current excluded
                    ds_intervals.append(
                        slice(interval_start_time, exc_start - step_length)
                    )
                    # Start next interval after end of excluded
                    interval_start_time = exc_end + step_length

                # Finish last interval
                ds_intervals.append(
                    slice(
                        interval_start_time, full_time_interval[-1].to_numpy()
                    )
                )

                # Check that all date slices are in correct order
                for time_slice in ds_intervals:
                    assert time_slice.start < time_slice.stop, (
                        "Can not make training subset from "
                        f"{time_slice.start} to {time_slice.stop}"
                    )

                # Create and concatenate all datasets
                self.train_dataset = torch.utils.data.ConcatDataset(
                    [
                        self.make_training_dataset(time_slice=time_slice)
                        for time_slice in ds_intervals
                    ]
                )
            else:
                self.train_dataset = self.make_training_dataset(time_slice=None)

            self.val_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                split="val",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
                interior_subsample_step=self.interior_subsample_step,
                boundary_subsample_step=self.boundary_subsample_step,
                dynamic_time_deltas=self.dynamic_time_deltas,
            )
            if self.eval_init_times:
                self.val_dataset = EvalSubsetWrapper(
                    self.val_dataset, self.eval_init_times
                )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                split=self.eval_split,
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
                interior_subsample_step=self.interior_subsample_step,
                boundary_subsample_step=self.boundary_subsample_step,
                dynamic_time_deltas=self.dynamic_time_deltas,
            )
            if self.eval_init_times:
                self.test_dataset = EvalSubsetWrapper(
                    self.test_dataset, self.eval_init_times
                )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )
