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
    standardize : bool, optional
        Whether to standardize the data. Default is True.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split="train",
        ar_steps=3,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        standardize=True,
    ):
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.datastore = datastore
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
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
                    f"({self.da_state.ensemble_member.size})",
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

    def _slice_state_time(self, da_state, idx, n_steps: int):
        """
        Produce a time slice of the given dataarray `da_state` (state) starting
        at `idx` and with `n_steps` steps. An `offset`is calculated based on the
        `num_past_forcing_steps` class attribute. `Offset` is used to offset the
        start of the sample, to assert that enough previous time steps are
        available for the 2 initial states and any corresponding forcings
        (calculated in `_slice_forcing_time`).

        Parameters
        ----------
        da_state : xr.DataArray
            The dataarray to slice. This is expected to have a `time` dimension
            if the datastore is providing analysis only data, and a
            `analysis_time` and `elapsed_forecast_duration` dimensions if the
            datastore is providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.

        Returns
        -------
        da_sliced : xr.DataArray
            The sliced dataarray with dims ('time', 'grid_index',
            'state_feature').
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
            da_sliced = da_state.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(start_idx, end_idx),
            )
            # create a new time dimension so that the produced sample has a
            # `time` dimension, similarly to the analysis only data
            da_sliced["time"] = (
                da_sliced.analysis_time + da_sliced.elapsed_forecast_duration
            )
            da_sliced = da_sliced.swap_dims(
                {"elapsed_forecast_duration": "time"}
            )
        else:
            # For analysis data we slice the time dimension directly. The offset
            # is only relevant for the very first (and last) samples in the
            # dataset.
            start_idx = idx + max(0, self.num_past_forcing_steps - init_steps)
            end_idx = (
                idx + max(init_steps, self.num_past_forcing_steps) + n_steps
            )
            da_sliced = da_state.isel(time=slice(start_idx, end_idx))
        return da_sliced

    def _slice_forcing_time(self, da_forcing, idx, n_steps: int):
        """
        Produce a time slice of the given dataarray `da_forcing` (forcing)
        starting at `idx` and with `n_steps` steps. An `offset` is calculated
        based on the `num_past_forcing_steps` class attribute. It is used to
        offset the start of the sample, to ensure that enough previous time
        steps are available for the forcing data. The forcing data is windowed
        around the current autoregressive time step to include the past and
        future forcings.

        Parameters
        ----------
        da_forcing : xr.DataArray
            The forcing dataarray to slice. This is expected to have a `time`
            dimension if the datastore is providing analysis only data, and a
            `analysis_time` and `elapsed_forecast_duration` dimensions if the
            datastore is providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.

        Returns
        -------
        da_concat : xr.DataArray
            The sliced dataarray with dims ('time', 'grid_index',
            'window', 'forcing_feature').
        """
        # The current implementation requires at least 2 time steps for the
        # initial state (see GraphCast). The forcing data is windowed around the
        # current autregressive time step. The two `init_steps` can also be used
        # as past forcings.
        init_steps = 2
        da_list = []

        if self.datastore.is_forecast:
            # This implies that the data will have both `analysis_time` and
            # `elapsed_forecast_duration` dimensions for forecasts. We for now
            # simply select an analysis time and the first `n_steps` forecast
            # times (given no offset). Note that this means that we get one
            # sample per forecast.
            # Add a 'time' dimension using the actual forecast times
            offset = max(init_steps, self.num_past_forcing_steps)
            for step in range(n_steps):
                start_idx = offset + step - self.num_past_forcing_steps
                end_idx = offset + step + self.num_future_forcing_steps

                current_time = (
                    da_forcing.analysis_time[idx]
                    + da_forcing.elapsed_forecast_duration[offset + step]
                )

                da_sliced = da_forcing.isel(
                    analysis_time=idx,
                    elapsed_forecast_duration=slice(start_idx, end_idx + 1),
                )

                da_sliced = da_sliced.rename(
                    {"elapsed_forecast_duration": "window"}
                )

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(len(da_sliced.window))
                )

                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

            # Concatenate the list of DataArrays along the 'time' dimension
            da_concat = xr.concat(da_list, dim="time")

        else:
            # For analysis data, we slice the time dimension directly. The
            # offset is only relevant for the very first (and last) samples in
            # the dataset.
            offset = idx + max(init_steps, self.num_past_forcing_steps)
            for step in range(n_steps):
                start_idx = offset + step - self.num_past_forcing_steps
                end_idx = offset + step + self.num_future_forcing_steps

                # Slice the data over the desired time window
                da_sliced = da_forcing.isel(time=slice(start_idx, end_idx + 1))

                da_sliced = da_sliced.rename({"time": "window"})

                # Assign the 'window' coordinate to be relative positions
                da_sliced = da_sliced.assign_coords(
                    window=np.arange(len(da_sliced.window))
                )

                # Add a 'time' dimension to keep track of steps using actual
                # time coordinates
                current_time = da_forcing.time[offset + step]
                da_sliced = da_sliced.expand_dims(
                    dim={"time": [current_time.values]}
                )

                da_list.append(da_sliced)

            # Concatenate the list of DataArrays along the 'time' dimension
            da_concat = xr.concat(da_list, dim="time")

        return da_concat

    def _build_item_dataarrays(self, idx):
        """
        Create the dataarrays for the initial states, target states and forcing
        data for the sample at index `idx`.

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
        da_target_times : xr.DataArray
            The dataarray for the target times.
        """
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
            da_state = self.da_state.isel(ensemble_member=i_ensemble)
        else:
            da_state = self.da_state

        if self.da_forcing is not None:
            if "ensemble_member" in self.da_forcing.dims:
                raise NotImplementedError(
                    "Ensemble member not yet supported for forcing data"
                )
            da_forcing = self.da_forcing
        else:
            da_forcing = None

        # handle time sampling in a way that is compatible with both analysis
        # and forecast data
        da_state = self._slice_state_time(
            da_state=da_state, idx=idx, n_steps=self.ar_steps
        )
        if da_forcing is not None:
            da_forcing_windowed = self._slice_forcing_time(
                da_forcing=da_forcing, idx=idx, n_steps=self.ar_steps
            )

        # load the data into memory
        da_state.load()
        if da_forcing is not None:
            da_forcing_windowed.load()

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

        return (
            da_init_states,
            da_target_states,
            da_forcing_windowed,
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

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # target_times: (ar_steps,)

        return init_states, target_states, forcing, target_times

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
        ar_steps_train=3,
        ar_steps_eval=25,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self._datastore = datastore
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        if num_workers > 0:
            # default to spawn for now, as the default on linux "fork" hangs
            # when using dask (which the npyfilesmeps datastore uses)
            self.multiprocessing_context = "spawn"
        else:
            self.multiprocessing_context = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                datastore=self._datastore,
                split="train",
                ar_steps=self.ar_steps_train,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
            )
            self.val_dataset = WeatherDataset(
                datastore=self._datastore,
                split="val",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                split="test",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                num_past_forcing_steps=self.num_past_forcing_steps,
                num_future_forcing_steps=self.num_future_forcing_steps,
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
