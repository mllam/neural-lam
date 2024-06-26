# Standard library
import datetime as dt
import glob
import os
import re
from pathlib import Path

# Third-party
import dask.delayed
import numpy as np
import torch
from xarray.core.dataarray import DataArray
import parse
import dask
import dask.array
import xarray as xr

# First-party

from ..base import BaseCartesianDatastore
from .config import NpyConfig

STATE_FILENAME_FORMAT = "nwp_{analysis_time:%Y%m%d%H}_mbr{member_id:03d}.npy"
TOA_SW_DOWN_FLUX_FILENAME_FORMAT = "nwp_toa_downwelling_shortwave_flux_{analysis_time:%Y%m%d%H}.npy"
COLUMN_WATER_FILENAME_FORMAT = "wtr_{analysis_time:%Y%m%d%H}.npy"



class NumpyFilesDatastore(BaseCartesianDatastore):
    __doc__ = f"""
    Represents a dataset stored as numpy files on disk. The dataset is assumed
    to be stored in a directory structure where each sample is stored in a
    separate file. The file-name format is assumed to be '{STATE_FILENAME_FORMAT}'
    
    The MEPS dataset is organised into three splits: train, val, and test. Each
    split has a set of files which are:

    - `{STATE_FILENAME_FORMAT}`:
        The state variables for a forecast started at `analysis_time` with
        member id `member_id`. The dimensions of the array are 
        `[forecast_timestep, y, x, feature]`.
        
    - `{TOA_SW_DOWN_FLUX_FILENAME_FORMAT}`:
        The top-of-atmosphere downwelling shortwave flux at `time`. The
        dimensions of the array are `[forecast_timestep, y, x]`.
        
    - `{COLUMN_WATER_FILENAME_FORMAT}`:
        The column water at `time`. The dimensions of the array are
        `[y, x]`.

    
    Folder structure:

    meps_example_reduced
    ├── data_config.yaml
    ├── samples
    │   ├── test
    │   │   ├── nwp_2022090100_mbr000.npy
    │   │   ├── nwp_2022090100_mbr001.npy
    │   │   ├── nwp_2022090112_mbr000.npy
    │   │   ├── nwp_2022090112_mbr001.npy
    │   │   ├── ...
    │   │   ├── nwp_toa_downwelling_shortwave_flux_2022090100.npy
    │   │   ├── nwp_toa_downwelling_shortwave_flux_2022090112.npy
    │   │   ├── ...
    │   │   ├── wtr_2022090100.npy
    │   │   ├── wtr_2022090112.npy
    │   │   └── ...
    │   ├── train
    │   │   ├── nwp_2022040100_mbr000.npy
    │   │   ├── nwp_2022040100_mbr001.npy
    │   │   ├── ...
    │   │   ├── nwp_2022040112_mbr000.npy
    │   │   ├── nwp_2022040112_mbr001.npy
    │   │   ├── ...
    │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy
    │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
    │   │   ├── ...
    │   │   ├── wtr_2022040100.npy
    │   │   ├── wtr_2022040112.npy
    │   │   └── ...
    │   └── val
    │       ├── nwp_2022060500_mbr000.npy
    │       ├── nwp_2022060500_mbr001.npy
    │       ├── ...
    │       ├── nwp_2022060512_mbr000.npy
    │       ├── nwp_2022060512_mbr001.npy
    │       ├── ...
    │       ├── nwp_toa_downwelling_shortwave_flux_2022060500.npy
    │       ├── nwp_toa_downwelling_shortwave_flux_2022060512.npy
    │       ├── ...
    │       ├── wtr_2022060500.npy
    │       ├── wtr_2022060512.npy
    │       └── ...
    └── static
        ├── border_mask.npy
        ├── diff_mean.pt
        ├── diff_std.pt
        ├── flux_stats.pt
        ├── grid_features.pt
        ├── nwp_xy.npy
        ├── parameter_mean.pt
        ├── parameter_std.pt
        ├── parameter_weights.npy
        └── surface_geopotential.npy
    
    For the MEPS dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_y = 268
    dim_x = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    
    For the MEPS reduced dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_y = 134
    dim_x = 119
    N_grid = 134x119 = 15946
    d_features = 8
    d_forcing = 1
    """
    is_ensemble = True

    def __init__(
        self,
        root_path,
    ):
        # XXX: This should really be in the config file, not hard-coded in this class
        self._num_timesteps = 65
        self._step_length = 3  # 3 hours
        self._num_ensemble_members = 2

        self.root_path = Path(root_path)
        self._config = NpyConfig.from_file(self.root_path / "data_config.yaml")
        pass
    
    def get_dataarray(self, category: str, split: str) -> DataArray:
        """
        Get the data array for the given category and split of data. If the category
        is 'state', the data array will be a concatenation of the data arrays for all
        ensemble members. The data will be loaded as a dask array, so that the data
        isn't actually loaded until it's needed.

        Parameters
        ----------
        category : str
            The category of the data to load. One of 'state', 'forcing', or 'static'.
        split : str
            The dataset split to load the data for. One of 'train', 'val', or 'test'.
            
        Returns
        -------
        xr.DataArray
            The data array for the given category and split, with dimensions per category:
            state:              `[time, analysis_time, grid_index, feature, ensemble_member]`
            forcing & static:   `[time, analysis_time, grid_index, feature]`
        """
        if category == "state":
            # for the state category, we need to load all ensemble members
            da = xr.concat(
                [
                    self._get_single_timeseries_dataarray(category=category, split=split, member=member)
                    for member in range(self._num_ensemble_members)
                ],
                dim="ensemble_member"
            )
        else:
            da = self._get_single_timeseries_dataarray(category=category, split=split)
        return da
    
    def _get_single_timeseries_dataarray(self, category: str, split: str, member: int = None) -> DataArray:
        """
        Get the data array spanning the complete time series for a given category and split
        of data. If the category is 'state', the member argument should be specified to select
        the ensemble member to load. The data will be loaded as a dask array, so that the data
        isn't actually loaded until it's needed.

        Parameters
        ----------
        category : str
            The category of the data to load. One of 'state', 'forcing', or 'static'.
        split : str
            The dataset split to load the data for. One of 'train', 'val', or 'test'.
        member : int, optional
            The ensemble member to load. Only applicable for the 'state' category.
        
        Returns
        -------
        xr.DataArray
            The data array for the given category and split, with dimensions
            `[time, analysis_time, grid_index, feature]` for all categories of data
        """
        assert split in ("train", "val", "test"), "Unknown dataset split"
        
        if member is not None and category != "state":
            raise ValueError("Member can only be specified for the 'state' category")
        
        # XXX: we here assume that the grid shape is the same for all categories
        grid_shape = self.grid_shape_state

        analysis_times = self._get_analysis_times(split=split)
        fp_split = self.root_path / "samples" / split
        
        file_dims = ["time", "y", "x", "feature"]
        elapsed_time = self.step_length * np.arange(self._num_timesteps) * np.timedelta64(1, "h")
        arr_shape = [len(elapsed_time)] + grid_shape
        coords = dict(
            analysis_time=analysis_times,
            time=elapsed_time,
            y=np.arange(grid_shape[0]),
            x=np.arange(grid_shape[1]),
        )
        
        extra_kwargs = {}
        add_feature_dim = False
        if category == "state":
            filename_format = STATE_FILENAME_FORMAT
            # only select one member for now
            extra_kwargs["member_id"] = member
            # state has multiple features
            num_state_variables = self.get_num_data_vars
            arr_shape += [num_state_variables]
            coords["feature"] = self.get_vars_names(category="state")
        elif category == "forcing":
            filename_format = TOA_SW_DOWN_FLUX_FILENAME_FORMAT
            arr_shape += [1]
            # XXX: this should really be saved in the data-config
            coords["feature"] = ["toa_downwelling_shortwave_flux"]
            add_feature_dim = True
        elif category == "static":
            filename_format = COLUMN_WATER_FILENAME_FORMAT
            arr_shape += [1]
            # XXX: this should really be saved in the data-config
            coords["feature"] = ["column_water"]
            add_feature_dim = True
        else:
            raise NotImplementedError(f"Category {category} not supported")
            
        filepaths = [
            fp_split / filename_format.format(analysis_time=analysis_time, **extra_kwargs)
            for analysis_time in analysis_times
        ]
        
        # use dask.delayed to load the numpy files, so that loading isn't
        # done until the data is actually needed
        @dask.delayed
        def _load_np(fp):
            arr = np.load(fp)
            if add_feature_dim:
                arr = arr[..., np.newaxis]
            return arr

        arrays = [
            dask.array.from_delayed(
               _load_np(fp), shape=arr_shape, dtype=np.float32
             ) for fp in filepaths
        ]
        
        arr_all = dask.array.stack(arrays, axis=0)
        
        da = xr.DataArray(
            arr_all,
            dims=["analysis_time"] + file_dims,
            coords=coords,
            name=category
        )
        
        # stack the [x, y] dimensions into a `grid_index` dimension
        da = da.stack(grid_index=["y", "x"])
        
        if category == "forcing":
            # add datetime forcing as a feature
            # to do this we create a forecast time variable which has the dimensions of
            # (analysis_time, time) with values that are the actual forecast time of each
            # time step. But calling .chunk({"time": 1}) this time variable is turned into
            # a dask array and so execution of the calculation is delayed until the feature
            # values are actually used.
            da_forecast_time = (da.time + da.analysis_time).chunk({"time": 1})
            da_datetime_forcing_features = self._calc_datetime_forcing_features(da_time=da_forecast_time)
            da = xr.concat([da, da_datetime_forcing_features], dim="feature")
        
        return da
        
    def _get_analysis_times(self, split):
        """
        Get the analysis times for the given split by parsing the filenames
        of all the files found for the given split.

        Parameters
        ----------
        split : str
            The dataset split to get the analysis times for.

        Returns
        -------
        List[dt.datetime]
            The analysis times for the given split.
        """
        pattern = re.sub(r'{analysis_time:[^}]*}', '*', STATE_FILENAME_FORMAT)
        pattern = re.sub(r'{member_id:[^}]*}', '*', pattern)
        
        sample_dir = self.root_path / "samples" / split
        sample_files = sample_dir.glob(pattern)
        times = []
        for fp in sample_files:
            name_parts = parse.parse(STATE_FILENAME_FORMAT, fp.name)
            times.append(name_parts["analysis_time"])
            
        return times
    
    def _calc_datetime_forcing_features(self, da_time: xr.DataArray):
        da_hour_angle = da_time.dt.hour / 12 * np.pi
        da_year_angle = da_time.dt.dayofyear / 365 * 2 * np.pi

        da_datetime_forcing = xr.concat(
            (
                np.sin(da_hour_angle),
                np.cos(da_hour_angle),
                np.sin(da_year_angle),
                np.cos(da_year_angle),
            ),
            dim="feature",
        )
        da_datetime_forcing = (da_datetime_forcing + 1) / 2  # Rescale to [0,1]
        da_datetime_forcing["feature"] = ["sin_hour", "cos_hour", "sin_year", "cos_year"]
        
        return da_datetime_forcing

    def get_vars_units(self, category: str) -> torch.List[str]:
        if category == "state":
            return self._config["dataset"]["var_units"]
        else:
            raise NotImplementedError(f"Category {category} not supported")
    
    def get_vars_names(self, category: str) -> torch.List[str]:
        if category == "state":
            return self._config["dataset"]["var_names"]
        else:
            raise NotImplementedError(f"Category {category} not supported")
        
    @property
    def get_num_data_vars(self) -> int:
        return len(self.get_vars_names(category="state"))
        
    def get_xy(self, category: str, stacked: bool) -> np.ndarray:
        arr = np.load(self.root_path / "static" / "nwp_xy.npy")
        
        assert arr.shape[0] == 2, "Expected 2D array"
        assert arr.shape[1:] == tuple(self.grid_shape_state), "Unexpected shape"
        
        if stacked:
            return arr
        else:
            return arr[0], arr[1]
    
    @property
    def step_length(self):
        return self._step_length
        
    @property
    def coords_projection(self):
        return self._config.coords_projection
    
    @property
    def grid_shape_state(self):
        return self._config.grid_shape_state
    
    @property
    def boundary_mask(self):
        return np.load(self.root_path / "static" / "border_mask.npy")