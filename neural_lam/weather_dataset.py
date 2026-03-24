# Standard library
import datetime
import warnings
from typing import Dict, List, Optional, Tuple, Union, Generator

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from loguru import logger

# First-party
from neural_lam.datastore.base import BaseDatastore


class WeatherDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing weather data from a given datastore.

    Args:
        datastore (BaseDatastore): The datastore to load the data from (e.g., mdp).
        split (str, optional): The data split to use ("train", "val", or "test"). 
            Defaults to "train".
        ar_steps (int, optional): Number of autoregressive steps. Defaults to 3.
        num_past_forcing_steps (int, optional): Number of past time steps to 
            include in forcing input. Defaults to 1.
        num_future_forcing_steps (int, optional): Number of future time steps to 
            include in forcing input. Defaults to 1.
        standardize (bool, optional): Whether to standardize the data. 
            Defaults to True.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split: str = "train",
        ar_steps: int = 3,
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        load_single_member: bool = False,
        standardize: bool = True,
    ) -> None:
        super().__init__()

        self.split = split
        self.ar_steps = ar_steps
        self.datastore = datastore
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.load_single_member = load_single_member

        self.da_state = self.datastore.get_dataarray(
            category="state", split=self.split
        )
        self.da_forcing = self.datastore.get_dataarray(
            category="forcing", split=self.split
        )

        if self.__len__() <= 0 and self.da_state is not None:
            raise ValueError(
                f"The provided datastore only provides {len(self.da_state.time)} "
                f"total time steps, which is too few for the `{split}` split. "
                "Try reducing ar_steps or forcing window size."
            )

        # Check the dimensions and their ordering
        parts = dict(state=self.da_state)
        if self.da_forcing is not None:
            parts["forcing"] = self.da_forcing

        for part, da in parts.items():
            expected_dim_order = self.datastore.expected_dim_order(category=part)
            if da is not None and da.dims != expected_dim_order:
                raise ValueError(
                    f"Dimension order of `{part}` ({da.dims}) does not match "
                    f"expected ({expected_dim_order})."
                )

        self.standardize = standardize
        if standardize:
            self.ds_state_stats = self.datastore.get_standardization_dataarray(
                category="state"
            )
            self.da_state_mean = self.ds_state_stats.state_mean
            self.da_state_std = self.ds_state_stats.state_std

            if self.da_forcing is not None:
                self.ds_forcing_stats = self.datastore.get_standardization_dataarray(
                    category="forcing"
                )
                self.da_forcing_mean = self.ds_forcing_stats.forcing_mean
                self.da_forcing_std = self.ds_forcing_stats.forcing_std
            else:
                self.da_forcing_mean = None
                self.da_forcing_std = None

            self.state_std_safe = self._compute_std_safe(
                self.da_state_std, "state"
            )
            self.forcing_std_safe = (
                self._compute_std_safe(self.da_forcing_std, "forcing")
                if self.da_forcing_std is not None else None
            )

    def _compute_std_safe(self, std: xr.DataArray, feature: str) -> xr.DataArray:
        """
        Ensures standard deviation is above machine epsilon to avoid division by zero.

        Args:
            std (xr.DataArray): The standard deviation array.
            feature (str): Name of the feature category for logging.

        Returns:
            xr.DataArray: The standard deviation array with near-zero values 
                replaced by epsilon.
        """
        eps = np.finfo(std.dtype).eps
        if bool((std <= eps).any()):
            logger.warning(
                f"Some {feature} features have near-zero std and will be "
                "standardized using machine epsilon to avoid NaN."
            )
        return std.where(std > eps, other=eps)

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset.

        Returns:
            int: Number of samples.
        """
        if self.datastore.is_forecast:
            if self.datastore.is_ensemble:
                warnings.warn(
                    "Only using first ensemble member; dataset size effectively "
                    "reduced.", UserWarning
                )

            n_forecast_steps = self.da_state.elapsed_forecast_duration.size
            if n_forecast_steps < 2 + self.ar_steps:
                raise ValueError(
                    f"Forecast steps ({n_forecast_steps}) < required "
                    f"(2 + {self.ar_steps})."
                )
            return self.da_state.analysis_time.size
        
        return (
            len(self.da_state.time)
            - self.ar_steps
            - max(2, self.num_past_forcing_steps)
            - self.num_future_forcing_steps
        )

    def _slice_state_time(self, da_state: xr.DataArray, idx: int, n_steps: int) -> xr.DataArray:
        """
        Produces a time slice of the state data.

        Args:
            da_state (xr.DataArray): DataArray to slice.
            idx (int): Index of the starting time step.
            n_steps (int): Number of steps to include.

        Returns:
            xr.DataArray: Sliced DataArray with dims ('time', 'grid_index', 'state_feature').
        """
        init_steps = 2
        if self.datastore.is_forecast:
            start_idx = max(0, self.num_past_forcing_steps - init_steps)
            end_idx = max(init_steps, self.num_past_forcing_steps) + n_steps
            da_sliced = da_state.isel(
                analysis_time=idx,
                elapsed_forecast_duration=slice(start_idx, end_idx),
            )
            da_sliced["time"] = (
                da_sliced.analysis_time + da_sliced.elapsed_forecast_duration
            )
            da_sliced = da_sliced.swap_dims({"elapsed_forecast_duration": "time"})
        else:
            start_idx = idx + max(0, self.num_past_forcing_steps - init_steps)
            end_idx = idx + max(init_steps, self.num_past_forcing_steps) + n_steps
            da_sliced = da_state.isel(time=slice(start_idx, end_idx))
        return da_sliced

    def _slice_forcing_time(self, da_forcing: xr.DataArray, idx: int, n_steps: int) -> xr.DataArray:
        """
        Produces a windowed time slice of the forcing data.

        Args:
            da_forcing (xr.DataArray): Forcing DataArray to slice.
            idx (int): Starting time step index.
            n_steps (int): Number of steps to include.

        Returns:
            xr.DataArray: Sliced array with dims ('time', 'grid_index', 'window', 'forcing_feature').
        """
        init_steps = 2
        da_list = []
        offset = (idx + max(init_steps, self.num_past_forcing_steps)) if not self.datastore.is_forecast else max(init_steps, self.num_past_forcing_steps)

        for step in range(n_steps):
            start_idx = offset + step - self.num_past_forcing_steps
            end_idx = offset + step + self.num_future_forcing_steps

            if self.datastore.is_forecast:
                current_time = da_forcing.analysis_time[idx] + da_forcing.elapsed_forecast_duration[offset + step]
                da_sliced = da_forcing.isel(
                    analysis_time=idx,
                    elapsed_forecast_duration=slice(start_idx, end_idx + 1),
                ).rename({"elapsed_forecast_duration": "window"})
            else:
                current_time = da_forcing.time[offset + step]
                da_sliced = da_forcing.isel(time=slice(start_idx, end_idx + 1)).rename({"time": "window"})

            da_sliced = da_sliced.assign_coords(window=np.arange(len(da_sliced.window)))
            da_sliced = da_sliced.expand_dims(dim={"time": [current_time.values]})
            da_list.append(da_sliced)

        return xr.concat(da_list, dim="time")

    def _build_item_dataarrays(self, idx: int) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Builds the underlying DataArrays for a single sample.

        Args:
            idx (int): The sample index.

        Returns:
            Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]: 
                Initial states, target states, forcing data, and target times.
        """
        if self.datastore.is_ensemble:
            warnings.warn("Only ensemble member 0 implemented.")
            da_state = self.da_state.isel(ensemble_member=0)
        else:
            da_state = self.da_state

        da_forcing = self.da_forcing
        if da_forcing is not None and "ensemble_member" in da_forcing.dims:
            raise NotImplementedError("Ensemble member not supported for forcing.")

        da_state = self._slice_state_time(da_state=da_state, idx=idx, n_steps=self.ar_steps)
        da_forcing_windowed = self._slice_forcing_time(da_forcing=da_forcing, idx=idx, n_steps=self.ar_steps) if da_forcing is not None else None

        da_state.load()
        if da_forcing_windowed is not None:
            da_forcing_windowed.load()

        da_init_states = da_state.isel(time=slice(0, 2))
        da_target_states = da_state.isel(time=slice(2, None))
        da_target_times = da_target_states.time

        if self.standardize:
            da_init_states = (da_init_states - self.da_state_mean) / self.state_std_safe
            da_target_states = (da_target_states - self.da_state_mean) / self.state_std_safe
            if da_forcing_windowed is not None:
                da_forcing_windowed = (da_forcing_windowed - self.da_forcing_mean) / self.forcing_std_safe

        if da_forcing_windowed is not None:
            da_forcing_windowed = da_forcing_windowed.stack(
                forcing_feature_windowed=("forcing_feature", "window")
            )
        else:
            da_forcing_windowed = xr.DataArray(
                data=np.empty((self.ar_steps, da_state.grid_index.size, 0)),
                dims=("time", "grid_index", "forcing_feature"),
                coords={"time": da_target_times, "grid_index": da_state.grid_index, "forcing_feature": []},
            )

        return da_init_states, da_target_states, da_forcing_windowed, da_target_times

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single training sample.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - init_states: (2, N_grid, d_features)
                - target_states: (ar_steps, N_grid, d_features)
                - forcing: (ar_steps, N_grid, d_windowed_forcing)
                - target_times: (ar_steps,)
        """
        (da_init, da_target, da_forcing, da_times) = self._build_item_dataarrays(idx=idx)

        tensor_dtype = torch.float32
        init_states = torch.tensor(da_init.values, dtype=tensor_dtype)
        target_states = torch.tensor(da_target.values, dtype=tensor_dtype)
        target_times = torch.tensor(
            da_times.astype("datetime64[ns]").astype("int64").values, dtype=torch.int64
        )
        forcing = torch.tensor(da_forcing.values, dtype=tensor_dtype)

        return init_states, target_states, forcing, target_times

    def __iter__(self) -> Generator:
        """Convenience method to iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]

    def create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[datetime.datetime, List[datetime.datetime]],
        category: str,
    ) -> xr.DataArray:
        """
        Constructs an xarray.DataArray from a torch.Tensor.

        Args:
            tensor (torch.Tensor): Tensor with shape (grid_index, feature) or 
                (time, grid_index, feature).
            time (Union[datetime.datetime, List[datetime.datetime]]): Time 
                coordinates for the tensor.
            category (str): Tensor category ("state", "forcing", or "static").

        Returns:
            xr.DataArray: Constructed DataArray with spatial coordinates.
        
        Raises:
            ValueError: If tensor dimensions and time list length mismatch.
        """
        def _is_listlike(obj):
            return hasattr(obj, "__iter__") and not isinstance(obj, str)

        add_time_as_dim = False
        if len(tensor.shape) == 2:
            dims = ["grid_index", f"{category}_feature"]
            if _is_listlike(time):
                raise ValueError(f"Expected single time for 2D tensor, got {len(time)}.")
        elif len(tensor.shape) == 3:
            add_time_as_dim = True
            dims = ["time", "grid_index", f"{category}_feature"]
            if not _is_listlike(time):
                raise ValueError("Expected list of times for 3D tensor.")
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {len(tensor.shape)}.")

        da_datastore_state = getattr(self, f"da_{category}")
        coords = {
            f"{category}_feature": da_datastore_state.state_feature,
            "grid_index": da_datastore_state.grid_index,
        }
        if add_time_as_dim:
            coords["time"] = time

        da = xr.DataArray(tensor.cpu().numpy(), dims=dims, coords=coords)

        for coord in ["x", "y"]:
            if coord in da_datastore_state.coords and coord not in da.coords:
                da.coords[coord] = da_datastore_state[coord]

        if not add_time_as_dim:
            da.coords["time"] = time

        return da


class WeatherDataModule(pl.LightningDataModule):
    """
    DataModule for organizing training, validation, and test weather datasets.

    Args:
        datastore (BaseDatastore): Datastore instance.
        ar_steps_train (int): Autoregressive steps for training. Defaults to 3.
        ar_steps_eval (int): Autoregressive steps for evaluation. Defaults to 25.
        standardize (bool): Whether to standardize data. Defaults to True.
        num_past_forcing_steps (int): Past forcing steps. Defaults to 1.
        num_future_forcing_steps (int): Future forcing steps. Defaults to 1.
        batch_size (int): Training batch size. Defaults to 4.
        num_workers (int): Number of worker processes for loading. Defaults to 16.
        eval_split (str): Split to use for evaluation. Defaults to "test".
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        ar_steps_train: int = 3,
        ar_steps_eval: int = 25,
        standardize: bool = True,
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        load_single_member: bool = False,
        batch_size: int = 4,
        num_workers: int = 16,
        eval_split: str = "test",
    ) -> None:
        super().__init__()
        self._datastore = datastore
        self.num_past_forcing_steps = num_past_forcing_steps
        self.num_future_forcing_steps = num_future_forcing_steps
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.load_single_member = load_single_member
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.eval_split = eval_split
        self.multiprocessing_context = "spawn" if num_workers > 0 else None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Initializes datasets for specified stage.

        Args:
            stage (str, optional): "fit", "test", or None.
        """
        common_kwargs = {
            "datastore": self._datastore,
            "standardize": self.standardize,
            "num_past_forcing_steps": self.num_past_forcing_steps,
            "num_future_forcing_steps": self.num_future_forcing_steps,
        }

        if stage in ("fit", None):
            self.train_dataset = WeatherDataset(split="train", ar_steps=self.ar_steps_train, **common_kwargs)
            self.val_dataset = WeatherDataset(split="val", ar_steps=self.ar_steps_eval, **common_kwargs)

        if stage in ("test", None):
            self.test_dataset = WeatherDataset(split=self.eval_split, ar_steps=self.ar_steps_eval, **common_kwargs)

    def _get_dataloader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        """Returns the training dataloader."""
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        """Returns the test dataloader."""
        return self._get_dataloader(self.test_dataset)