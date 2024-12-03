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


class WeatherDataset(torch.utils.data.Dataset):
    """Dataset class for weather data.

    This class loads and processes weather data from a given datastore.

    Parameters
    ----------
    datastore : BaseDatastore
        The datastore to load the data from (e.g. mdp).
    datastore_boundary : BaseDatastore
        The boundary datastore to load the data from (e.g. mdp).
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
    standardize : bool, optional
        Whether to standardize the data. Default is True.
    """

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
        standardize=True,
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
            self.da_boundary = self.datastore_boundary.get_dataarray(
                category="forcing", split=self.split
            )
        else:
            self.da_boundary = None

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
        else:
            self.da_state = self.da_state

        # Check time step consistency in state data
        if self.datastore.is_forecast:
            state_times = self.da_state.analysis_time
        else:
            state_times = self.da_state.time
        _ = self._get_time_step(state_times)

        # Check time coverage for forcing and boundary data
        if self.da_forcing is not None or self.da_boundary is not None:
            if self.datastore.is_forecast:
                state_times = self.da_state.analysis_time
            else:
                state_times = self.da_state.time
            state_time_min = state_times.min().values
            state_time_max = state_times.max().values

            if self.da_forcing is not None:
                # Forcing data is part of the same datastore as state data
                # During creation the time dimension of the forcing data
                # is matched to the state data
                if self.datastore.is_forecast:
                    forcing_times = self.da_forcing.analysis_time
                else:
                    forcing_times = self.da_forcing.time
                self._get_time_step(forcing_times.values)

            if self.da_boundary is not None:
                # Boundary data is part of a separate datastore
                # The boundary data is allowed to have a different time_step
                # Check that the boundary data covers the required time range
                if self.datastore_boundary.is_forecast:
                    boundary_times = self.da_boundary.analysis_time
                else:
                    boundary_times = self.da_boundary.time
                boundary_time_step = self._get_time_step(boundary_times.values)
                boundary_time_min = boundary_times.min().values
                boundary_time_max = boundary_times.max().values

                # Calculate required bounds for boundary using its time step
                boundary_required_time_min = (
                    state_time_min
                    - self.num_past_forcing_steps * boundary_time_step
                )
                boundary_required_time_max = (
                    state_time_max
                    + self.num_future_forcing_steps * boundary_time_step
                )

                if boundary_time_min > boundary_required_time_min:
                    raise ValueError(
                        f"Boundary data starts too late."
                        f"Required start: {boundary_required_time_min}, "
                        f"but boundary starts at {boundary_time_min}."
                    )

                if boundary_time_max < boundary_required_time_max:
                    raise ValueError(
                        f"Boundary data ends too early."
                        f"Required end: {boundary_required_time_max}, "
                        f"but boundary ends at {boundary_time_max}."
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
            if self.da_boundary is not None:
                self.ds_boundary_stats = (
                    self.datastore_boundary.get_standardization_dataarray(
                        category="forcing"
                    )
                )
                self.da_boundary_mean = self.ds_boundary_stats.forcing_mean
                self.da_boundary_std = self.ds_boundary_stats.forcing_std

    def __len__(self):
        if self.datastore.is_forecast:
            # for now we simply create a single sample for each analysis time
            # and then take the first (2 + ar_steps) forecast times. In
            # addition we only use the first ensemble member (if ensemble data
            # has been provided).
            # This means that for each analysis time we get a single sample

            if self.datastore.is_ensemble:
                warnings.warn(
                    "only using first ensemble member, so dataset size is "
                    " effectively reduced by the number of ensemble members "
                    f"({self.datastore._num_ensemble_members})",
                    UserWarning,
                )

            # check that there are enough forecast steps available to create
            # samples given the number of autoregressive steps requested
            n_forecast_steps = self.da_state.elapsed_forecast_duration.size
            if n_forecast_steps < 2 + self.ar_steps:
                raise ValueError(
                    "The number of forecast steps available "
                    f"({n_forecast_steps}) is less than the required "
                    f"2+ar_steps (2+{self.ar_steps}={2 + self.ar_steps}) for "
                    "creating a sample with initial and target states."
                )

            return self.da_state.analysis_time.size
        else:
            # Calculate the number of samples in the dataset n_samples = total
            # time steps - (autoregressive steps + past forcing + future
            # forcing)
            #:
            # Where:
            #   - total time steps: len(self.da_state.time)
            #   - autoregressive steps: self.ar_steps
            #   - past forcing: max(2, self.num_past_forcing_steps) (at least 2
            #     time steps are required for the initial state)
            #   - future forcing: self.num_future_forcing_steps
            return (
                len(self.da_state.time)
                - self.ar_steps
                - max(2, self.num_past_forcing_steps)
                - self.num_future_forcing_steps
            )

    def _get_time_step(self, times):
        """Calculate the time step from the data

        Parameters
        ----------
        times : xr.DataArray
            The time dataarray to calculate the time step from.

        Returns
        -------
        time_step : float
            The time step in the the format of the times dataarray.
        """
        time_diffs = np.diff(times)
        if not np.all(time_diffs == time_diffs[0]):
            raise ValueError(
                "Inconsistent time steps in data. "
                f"Found different time steps: {np.unique(time_diffs)}"
            )
        return time_diffs[0]

    def _slice_time(
        self,
        da_state,
        idx,
        n_steps: int,
        da_forcing_boundary=None,
        num_past_steps=None,
        num_future_steps=None,
    ):
        """
        Produce time slices of the given dataarrays `da_state` (state) and
        `da_forcing_boundary`. For the state data, slicing is done
        based on `idx`. For the forcing/boundary data, nearest neighbor matching
        is performed based on the state times. Additionally, the time difference
        between the matched forcing/boundary times and state times (in multiples
        of state time steps) is added to the forcing dataarray. This will be
        used as an additional input feature in the model (temporal embedding).

        Parameters
        ----------
        da_state : xr.DataArray
            The state dataarray to slice.
        idx : int
            The index of the time step to start the sample from in the state
            data.
        n_steps : int
            The number of time steps to include in the sample.
        da_forcing_boundary : xr.DataArray
            The forcing/boundary dataarray to slice.
        num_past_steps : int, optional
            The number of past time steps to include in the forcing/boundary
            data. Default is `None`.
        num_future_steps : int, optional
            The number of future time steps to include in the forcing/boundary
            data. Default is `None`.

        Returns
        -------
        da_state_sliced : xr.DataArray
            The sliced state dataarray with dims ('time', 'grid_index',
            'state_feature').
        da_forcing_boundary_matched : xr.DataArray
            The sliced state dataarray with dims ('time', 'grid_index',
            'forcing/boundary_feature_windowed').
            If no forcing/boundary data is provided, this will be `None`.
        """
        # The current implementation requires at least 2 time steps for the
        # initial state (see GraphCast).
        init_steps = 2
        # slice the dataarray to include the required number of time steps
        if self.datastore.is_forecast:
            start_idx = max(0, self.num_past_forcing_steps - init_steps)
            end_idx = max(init_steps, self.num_past_forcing_steps) + n_steps
            # this implies that the data will have both `analysis_time` and
            # `elapsed_forecast_duration` dimensions for forecasts. We for now
            # simply select a analysis time and the first `n_steps` forecast
            # times (given no offset). Note that this means that we get one
            # sample per forecast, always starting at forecast time 2.
            da_state_sliced = da_state.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(start_idx, end_idx),
            )
            # create a new time dimension so that the produced sample has a
            # `time` dimension, similarly to the analysis only data
            da_state_sliced["time"] = (
                da_state_sliced.analysis_time
                + da_state_sliced.elapsed_forecast_duration
            )
            da_state_sliced = da_state_sliced.swap_dims(
                {"elapsed_forecast_duration": "time"}
            )
            # Asserting that the forecast time step is consistent
            # In init this was only done for the analysis_time
            self._get_time_step(da_state_sliced.time)

        else:
            # For analysis data we slice the time dimension directly. The offset
            # is only relevant for the very first (and last) samples in the
            # dataset.
            start_idx = idx + max(0, num_past_steps - init_steps)
            end_idx = idx + max(init_steps, num_past_steps) + n_steps
            da_state_sliced = da_state.isel(time=slice(start_idx, end_idx))

        if da_forcing_boundary is None:
            return da_state_sliced, None

        # Get the state times and its temporal resolution for matching with
        # forcing data. No need to self._get_time_step as we have already
        # checked the time step consistency in the state data.
        state_times = da_state_sliced["time"]
        state_time_step = state_times.values[1] - state_times.values[0]

        # Here we cannot check 'self.datastore.is_forecast' directly because we
        # might be dealing with a datastore_boundary
        if "analysis_time" in da_forcing_boundary.dims:
            # Select the closest analysis time in the forcing/boundary data
            # This is mostly relevant for boundary data where the time steps
            # are not necessarily the same as the state data. But still fast
            # enough for forcing data where the time steps are the same.
            idx = np.abs(
                da_forcing_boundary.analysis_time.values
                - self.da_state.analysis_time.values[idx]
            ).argmin()
            # Add a 'time' dimension using the actual forecast times
            offset = max(init_steps, num_past_steps)
            da_list = []
            for step in range(n_steps):
                start_idx = offset + step - num_past_steps
                end_idx = offset + step + num_future_steps

                current_time = (
                    da_forcing_boundary.analysis_time[idx]
                    + da_forcing_boundary.elapsed_forecast_duration[
                        offset + step
                    ]
                )

                da_sliced = da_forcing_boundary.isel(
                    analysis_time=idx,
                    elapsed_forecast_duration=slice(start_idx, end_idx + 1),
                )

                da_sliced = da_sliced.rename(
                    {"elapsed_forecast_duration": "window"}
                )

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(-num_past_steps, num_future_steps + 1)
                )

                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

            # Generate temporal embedding `time_diff_steps` for the
            # forcing/boundary data. This is the time difference in multiples
            # of state time steps between the forcing/boundary time and the
            # state time.
            da_forcing_boundary_matched = xr.concat(da_list, dim="time")
            forcing_time_step = (
                da_forcing_boundary_matched.time.values[1]
                - da_forcing_boundary_matched.time.values[0]
            )
            # Since all time, grid_index and forcing_features share the same
            # temporal_embedding we can just use the first one
            da_forcing_boundary_matched["window"] = da_forcing_boundary_matched[
                "window"
            ] * (forcing_time_step / state_time_step)
            time_diff_steps = da_forcing_boundary_matched.isel(
                grid_index=0, forcing_feature=0, time=0
            ).window.values

        else:
            # For analysis data, we slice the time dimension directly. The
            # offset is only relevant for the very first (and last) samples in
            # the dataset.
            forcing_times = da_forcing_boundary["time"]

            # Compute time differences between forcing and state times
            # (in multiples of state time steps)
            # Retrieve the indices of the closest times in the forcing data
            time_deltas = (
                forcing_times.values[:, np.newaxis]
                - state_times.values[np.newaxis, :]
            ) / state_time_step
            idx_min = np.abs(time_deltas).argmin(axis=0)

            # Get the time differences for windowed time steps - they are
            # used as temporal embeddings and concatenated to the forcing
            # features later. All features share the same temporal embedding
            time_diff_steps = time_deltas[
                idx_min[init_steps]
                - num_past_steps : idx_min[init_steps]
                + num_future_steps
                + 1,
                init_steps,
            ]

            # Create window dimension for forcing data to stack later
            window_size = num_past_steps + num_future_steps + 1
            da_forcing_boundary_windowed = da_forcing_boundary.rolling(
                time=window_size, center=False
            ).construct(window_dim="window")
            da_forcing_boundary_matched = da_forcing_boundary_windowed.isel(
                time=idx_min[init_steps:]
            )

        # Add time difference as a new coordinate to concatenate to the
        # forcing features later as temporal embedding
        da_forcing_boundary_matched["time_diff_steps"] = (
            ("window"),
            time_diff_steps,
        )

        return da_state_sliced, da_forcing_boundary_matched

    def _process_windowed_data(self, da_windowed, da_state, da_target_times):
        """Helper function to process windowed data. This function stacks the
        'forcing_feature' and 'window' dimensions and adds the time step
        differences to the existing features as a temporal embedding.

        Parameters
        ----------
        da_windowed : xr.DataArray
            The windowed data to process. Can be `None` if no data is provided.
        da_state : xr.DataArray
            The state dataarray.
        da_target_times : xr.DataArray
            The target times.

        Returns
        -------
        da_windowed : xr.DataArray
            The processed windowed data. If `da_windowed` is `None`, an empty
            DataArray with the correct dimensions and coordinates is returned.

        """
        stacked_dim = "forcing_feature_windowed"
        if da_windowed is not None:
            # Stack the 'feature' and 'window' dimensions and add the
            # time step differences to the existing features as a temporal
            # embedding
            da_windowed = da_windowed.stack(
                {stacked_dim: ("forcing_feature", "window")}
            )
            # All data variables share the same temporal embedding, hence
            # only the first one is used
            da_windowed = xr.concat(
                [da_windowed, da_windowed.time_diff_steps],
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

        if self.da_boundary is not None:
            da_boundary = self.da_boundary
        else:
            da_boundary = None

        # if da_forcing_boundary is None, the function will return None for
        # da_forcing_windowed
        if da_boundary is not None:
            _, da_boundary_windowed = self._slice_time(
                da_state=da_state,
                idx=idx,
                n_steps=self.ar_steps,
                da_forcing_boundary=da_boundary,
                num_future_steps=self.num_future_boundary_steps,
                num_past_steps=self.num_past_boundary_steps,
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
            da_forcing_boundary=da_forcing,
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
        # and adds the time step differences as a temporal embedding.
        # It can handle `None` inputs for the forcing and boundary data
        # (and simlpy return an empty DataArray in that case).
        da_forcing_windowed = self._process_windowed_data(
            da_forcing_windowed, da_state, da_target_times
        )
        da_boundary_windowed = self._process_windowed_data(
            da_boundary_windowed, da_state, da_target_times
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
        datastore_boundary: BaseDatastore,
        ar_steps_train=3,
        ar_steps_eval=25,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=1,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self._datastore = datastore
        self._datastore_boundary = datastore_boundary
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.num_past_boundary_steps = num_past_boundary_steps
        self.num_future_boundary_steps = num_future_boundary_steps
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if num_workers > 0:
            # BUG: There also seem to be issues with "spawn" and `gloo`, to be
            # investigated. Defaults to spawn for now, as the default on linux
            # "fork" hangs when using dask (which the npyfilesmeps datastore
            # uses)
            self.multiprocessing_context = "spawn"
        else:
            self.multiprocessing_context = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                split="train",
                ar_steps=self.ar_steps_train,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
            )
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
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=self._datastore_boundary,
                split="test",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
                num_past_boundary_steps=self.num_past_boundary_steps,
                num_future_boundary_steps=self.num_future_boundary_steps,
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
