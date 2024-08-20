# Standard library
import warnings

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

        self.da_state = self.datastore.get_dataarray(category="state", split=self.split)
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )
        self.forcing_window_size = forcing_window_size

        # check that with the provided data-arrays and ar_steps that we have a
        # non-zero amount of samples
        if self.__len__() <= 0:
            raise ValueError(
                f"The provided datastore only provides {len(self.da_state.time)} "
                f"time steps for `{split}` split, which is less than the "
                f"required 2+ar_steps (2+{self.ar_steps}={2+self.ar_steps}) "
                "for creating a sample with initial and target states."
            )

        # Set up for standardization
        # TODO: This will become part of ar_model.py soon!
        self.standardize = standardize
        if standardize:
            self.ds_state_stats = self.datastore.get_normalization_dataarray(
                category="state"
            )

            self.da_state_mean = self.ds_state_stats.state_mean
            self.da_state_std = self.ds_state_stats.state_std

            if self.da_forcing is not None:
                self.ds_forcing_stats = self.datastore.get_normalization_dataarray(
                    category="forcing"
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
            # XXX: we should maybe check that the 2+ar_steps actually fits
            # in the elapsed_forecast_duration dimension, should that be checked here?
            return self.da_state.analysis_time.size
        else:
            # sample_len = 2 + ar_steps  (2 initial states + ar_steps target states)
            # n_samples = len(self.da_state.time) - sample_len + 1
            #           = len(self.da_state.time) - 2 - ar_steps + 1
            #           = len(self.da_state.time) - ar_steps - 1
            return len(self.da_state.time) - self.ar_steps - 1

    def _sample_time(self, da, idx, n_steps: int, n_timesteps_offset: int = 0):
        """Produce a time
        slice of the given
        dataarray `da` (state
        or forcing) starting
        at `idx` and with
        `n_steps` steps. The
        `n_timesteps_offset`
        parameter is used to
        offset the start of
        the sample, for
        example to exclude the
        first two steps when
        sampling the forcing
        data (and to produce
        the windowing samples
        of forcing data by
        increasing the offset
        for each window).

        Parameters
        ----------
        da : xr.DataArray
            The dataarray to sample from. This is expected to have a `time`
            dimension if the datastore is providing analysis only data, and a
            `analysis_time` and `elapsed_forecast_duration` dimensions if the
            datastore is providing forecast data.
        idx : int
            The index of the time step to start the sample from.
        n_steps : int
            The number of time steps to include in the sample.

        """
        # selecting the time slice
        if self.datastore.is_forecast:
            # this implies that the data will have both `analysis_time` and
            # `elapsed_forecast_duration` dimensions for forecasts we for now
            # simply select a analysis time and then the next ar_steps forecast
            # times
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
                time=slice(idx + n_timesteps_offset, idx + n_steps + n_timesteps_offset)
            )
        return da

    def __getitem__(self, idx):
        """Return a single training sample, which consists of the initial states, target
        states, forcing and batch times.

        The implementation currently uses xarray.DataArray objects for the
        normalisation so that we can make us of xarray's broadcasting
        capabilities. This makes it possible to normalise with both global
        means, but also for example where a grid-point mean has been computed.
        This code will have to be replace if normalisation is to be done on the
        GPU to handle different shapes of the normalisation.

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
        # handling ensemble data
        if self.datastore.is_ensemble:
            # for the now the strategy is to simply select a random ensemble member
            # XXX: this could be changed to include all ensemble members by
            # splitting `idx` into two parts, one for the analysis time and one
            # for the ensemble member and then increasing self.__len__ to
            # include all ensemble members
            i_ensemble = np.random.randint(self.da_state.ensemble_member.size)
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
        da_state = self._sample_time(da=da_state, idx=idx, n_steps=2 + self.ar_steps)

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

        batch_times = da_target_states.time.values.astype(float)

        if self.standardize:
            da_init_states = (da_init_states - self.da_state_mean) / self.da_state_std
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

        init_states = torch.tensor(da_init_states.values, dtype=torch.float32)
        target_states = torch.tensor(da_target_states.values, dtype=torch.float32)

        if self.da_forcing is None:
            # create an empty forcing tensor
            forcing = torch.empty(
                (self.ar_steps, da_state.grid_index.size, 0),
                dtype=torch.float32,
            )
        else:
            forcing = torch.tensor(da_forcing_windowed.values, dtype=torch.float32)

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps, N_grid, d_features)
        # forcing: (ar_steps, N_grid, d_windowed_forcing)
        # batch_times: (ar_steps,)

        return init_states, target_states, forcing, batch_times

    def __iter__(self):
        """Convenience method to iterate over the dataset.

        This isn't used by pytorch DataLoader which itself implements an iterator that
        uses Dataset.__getitem__ and Dataset.__len__.

        """
        for i in range(len(self)):
            yield self[i]


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
            shuffle=False,
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
