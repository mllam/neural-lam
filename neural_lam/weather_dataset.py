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

    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split="train",
        ar_steps=3,
        forcing_window_size=3,
        standardize=True,
    ):
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.datastore = datastore

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )
        self.forcing_window_size = forcing_window_size

        # check that with the provided data-arrays and ar_steps that we have a
        # non-zero amount of samples
        if self.__len__() <= 0:
            raise ValueError(
                "The provided datastore only provides "
                f"{len(self.da_state.time)} time steps for `{split}` split, "
                f"which is less than the required 2+ar_steps "
                f"(2+{self.ar_steps}={2 + self.ar_steps}) for creating a "
                "sample with initial and target states."
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
            # and then the next ar_steps forecast times
            if self.datastore.is_ensemble:
                warnings.warn(
                    "only using first ensemble member, so dataset size is "
                    " effectively reduced by the number of ensemble members "
                    f"({self.da_state.ensemble_member.size})",
                    UserWarning,
                )
            # XXX: we should maybe check that the 2+ar_steps actually fits in
            # the elapsed_forecast_duration dimension, should that be checked
            # here?
            return self.da_state.analysis_time.size
        else:
            # sample_len = 2 + ar_steps
            #             (2 initial states + ar_steps target states)
            # n_samples = len(self.da_state.time) - sample_len + 1
            #           = len(self.da_state.time) - 2 - ar_steps + 1
            #           = len(self.da_state.time) - ar_steps - 1
            return len(self.da_state.time) - self.ar_steps - 1

    def _sample_time(self, da, idx, n_steps: int, n_timesteps_offset: int = 0):
        """
        Produce a time slice of the given dataarray `da` (state or forcing)
        starting at `idx` and with `n_steps` steps. The `n_timesteps_offset`
        parameter is used to offset the start of the sample, for example to
        exclude the first two steps when sampling the forcing data (and to
        produce the windowing samples of forcing data by increasing the offset
        for each window).

        Parameters
        ----------
        da : xr.DataArray
            The dataarray to slice. This is expected to have a `time`
            dimension if the datastore is providing analysis only data, and a
            `analysis_time` and `elapsed_forecast_duration` dimensions if the
            datastore is providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.
        n_timestep_offset : int
            A number of timesteps to use as offset from the start time of the slice
        """
        # selecting the time slice
        if self.datastore.is_forecast:
            # this implies that the data will have both `analysis_time` and
            # `elapsed_forecast_duration` dimensions for forecasts. We for now
            # simply select a analysis time and the first `n_steps` forecast
            # times (given no offset). Note that this means that we get one
            # sample per forecast, always starting at forecast time 2.
            da = da.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(
                    n_timesteps_offset, n_steps + n_timesteps_offset
                ),
            )
            # create a new time dimension so that the produced sample has a
            # `time` dimension, similarly to the analysis only data
            da["time"] = da.analysis_time + da.elapsed_forecast_duration
            da = da.swap_dims({"elapsed_forecast_duration": "time"})
        else:
            # only `time` dimension for analysis only data
            da = da.isel(
                time=slice(
                    idx + n_timesteps_offset, idx + n_steps + n_timesteps_offset
                )
            )
        return da

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
        da_state = self._sample_time(
            da=da_state, idx=idx, n_steps=2 + self.ar_steps
        )

        if da_forcing is not None:
            das_forcing = []
            for n in range(self.forcing_window_size):
                da_ = self._sample_time(
                    da=da_forcing,
                    idx=idx,
                    n_steps=self.ar_steps,
                    n_timesteps_offset=n,
                )
                if n > 0:
                    da_ = da_.drop_vars("time")
                das_forcing.append(da_)
            da_forcing_windowed = xr.concat(das_forcing, dim="window_sample")

        # load the data into memory
        da_state = da_state.load()
        if da_forcing is not None:
            da_forcing_windowed = da_forcing_windowed.load()

        # ensure the dimensions are in the correct order
        da_state = da_state.transpose("time", "grid_index", "state_feature")

        if da_forcing is not None:
            da_forcing_windowed = da_forcing_windowed.transpose(
                "time", "grid_index", "forcing_feature", "window_sample"
            )

        da_init_states = da_state.isel(time=slice(None, 2))
        da_target_states = da_state.isel(time=slice(2, None))

        da_target_times = da_target_states.time

        if self.standardize:
            da_init_states = (
                da_init_states - self.da_state_mean
            ) / self.da_state_std
            da_target_states = (
                da_target_states - self.da_state_mean
            ) / self.da_state_std

            if self.da_forcing is not None:
                da_forcing_windowed = (
                    da_forcing_windowed - self.da_forcing_mean
                ) / self.da_forcing_std

        if self.da_forcing is not None:
            # stack the `forcing_feature` and `window_sample` dimensions into a
            # single `forcing_feature` dimension
            da_forcing_windowed = da_forcing_windowed.stack(
                forcing_feature_windowed=("forcing_feature", "window_sample")
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
            (i.e. time, grid_index, {category}_feature).
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
            tensor.numpy(),
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
        forcing_window_size=3,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self._datastore = datastore
        self.forcing_window_size = forcing_window_size
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
            # when using dask (which the npyfiles datastore uses)
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
                forcing_window_size=self.forcing_window_size,
            )
            self.val_dataset = WeatherDataset(
                datastore=self._datastore,
                split="val",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                forcing_window_size=self.forcing_window_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                datastore=self._datastore,
                split="test",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                forcing_window_size=self.forcing_window_size,
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
