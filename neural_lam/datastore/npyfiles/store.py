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
    is_forecast = True

    def __init__(
        self,
        root_path,
    ):
        # XXX: This should really be in the config file, not hard-coded in this class
        self._num_timesteps = 65
        self._step_length = 3  # 3 hours
        self._num_ensemble_members = 2

        self.root_path = Path(root_path)
        self.config = NpyConfig.from_file(self.root_path / "data_config.yaml")
    
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
            state:     `[time, analysis_time, grid_index, feature, ensemble_member]`
            forcing:   `[time, analysis_time, grid_index, feature]`
            static:    `[grid_index, feature]`
        """
        if category == "state":
            das = []
            # for the state category, we need to load all ensemble members
            for member in range(self._num_ensemble_members):
                da_member = self._get_single_timeseries_dataarray(features=self.get_vars_names(category="state"), split=split, member=member)
                das.append(da_member)
            da = xr.concat(das, dim="ensemble_member")

        elif category == "forcing":
            # the forcing features are in separate files, so we need to load them separately
            features = ["toa_downwelling_shortwave_flux", "column_water"]
            das = [self._get_single_timeseries_dataarray(features=[feature], split=split) for feature in features]
            da = xr.concat(das, dim="feature")
        
        elif category == "static":
            # the static features are collected in three files:
            # - surface_geopotential
            # - border_mask
            # - x, y
            das = []
            for features in [["surface_geopotential"], ["border_mask"], ["x", "y"]]:
                da = self._get_single_timeseries_dataarray(features=features, split=split)
                das.append(da)
            da = xr.concat(das, dim="feature").transpose("grid_index", "feature")

        else:
            raise NotImplementedError(category)
        
        da = da.rename(dict(feature=f"{category}_feature"))
        
        if category == "forcing":
            # add datetime forcing as a feature
            # to do this we create a forecast time variable which has the dimensions of
            # (analysis_time, elapsed_forecast_time) with values that are the actual forecast time of each
            # time step. By calling .chunk({"elapsed_forecast_time": 1}) this time variable is turned into
            # a dask array and so execution of the calculation is delayed until the feature
            # values are actually used.
            da_forecast_time = (da.analysis_time + da.elapsed_forecast_time).chunk({"elapsed_forecast_time": 1})
            da_datetime_forcing_features = self._calc_datetime_forcing_features(da_time=da_forecast_time)
            da = xr.concat([da, da_datetime_forcing_features], dim=f"{category}_feature")
            
        da.name = category
        
        # check that we have the right features
        actual_features = list(da[f"{category}_feature"].values)
        expected_features = self.get_vars_names(category=category)
        if actual_features != expected_features:
            raise ValueError(f"Expected features {expected_features}, got {actual_features}")
        
        return da
    
    def _get_single_timeseries_dataarray(self, features: str, split: str, member: int = None) -> DataArray:
        """
        Get the data array spanning the complete time series for a given set of features and split
        of data. If the category is 'state', the member argument should be specified to select
        the ensemble member to load. The data will be loaded using dask.delayed, so that the data
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
            `[elapsed_forecast_time, analysis_time, grid_index, feature]` for all categories of data
        """
        assert split in ("train", "val", "test"), "Unknown dataset split"
        
        if member is not None and features != self.get_vars_names(category="state"):
            raise ValueError("Member can only be specified for the 'state' category")
        
        # XXX: we here assume that the grid shape is the same for all categories
        grid_shape = self.grid_shape_state

        fp_samples = self.root_path / "samples" / split
        
        file_params = {}
        add_feature_dim = False
        features_vary_with_analysis_time = True
        if features == self.get_vars_names(category="state"):
            filename_format = STATE_FILENAME_FORMAT
            file_dims = ["elapsed_forecast_time", "y", "x", "feature"]
            # only select one member for now
            file_params["member_id"] = member
        elif features == ["toa_downwelling_shortwave_flux"]:
            filename_format = TOA_SW_DOWN_FLUX_FILENAME_FORMAT
            file_dims = ["elapsed_forecast_time", "y", "x", "feature"]
            add_feature_dim = True
        elif features == ["column_water"]:
            filename_format = COLUMN_WATER_FILENAME_FORMAT
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
        elif features == ["surface_geopotential"]:
            filename_format = "surface_geopotential.npy"
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
            features_vary_with_analysis_time = False
            # XXX: surface_geopotential is the same for all splits, and so saved in static/
            fp_samples = self.root_path / "static"
            import ipdb; ipdb.set_trace()
        elif features == ["border_mask"]:
            filename_format = "border_mask.npy"
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
            features_vary_with_analysis_time = False
            # XXX: border_mask is the same for all splits, and so saved in static/
            fp_samples = self.root_path / "static"
        elif features == ["x", "y"]:
            filename_format = "nwp_xy.npy"
            file_dims = ["y", "x", "feature"]
            features_vary_with_analysis_time = False
            # XXX: x, y are the same for all splits, and so saved in static/
            fp_samples = self.root_path / "static"
        else:
            raise NotImplementedError(f"Reading of variables set `{features}` not supported")
        
        if features_vary_with_analysis_time:
            dims = ["analysis_time"] + file_dims
        else:
            dims = file_dims
        
        coords = {}
        arr_shape = []
        for d in dims:
            if d == "elapsed_forecast_time":
                coord_values = self.step_length * np.arange(self._num_timesteps) * np.timedelta64(1, "h")
            elif d == "analysis_time":
                coord_values = self._get_analysis_times(split=split)
            elif d == "y":
                coord_values = np.arange(grid_shape[0])
            elif d == "x":
                coord_values = np.arange(grid_shape[1])
            elif d == "feature":
                coord_values = features
            else:
                raise NotImplementedError(f"Dimension {d} not supported")
            
            print(f"{d}: {len(coord_values)}")
            
            coords[d] = coord_values
            if d != "analysis_time":
                # analysis_time varies across the different files, but not within a single file
                arr_shape.append(len(coord_values))
                
        print(f"{features}: {dims=} {file_dims=} {arr_shape=}")
            
        if features_vary_with_analysis_time:
            filepaths = [
                fp_samples / filename_format.format(analysis_time=analysis_time, **file_params)
                for analysis_time in coords["analysis_time"]
            ]
        else:
            filepaths = [fp_samples / filename_format.format(**file_params)]
        
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
        
        if features_vary_with_analysis_time:
            arr_all = dask.array.stack(arrays, axis=0)
        else:
            arr_all = arrays[0]
        
        # if features == ["column_water"]:
        #     # for column water, we need to repeat the array for each forecast time
        #     # first insert a new axis for the forecast time
        #     arr_all = np.expand_dims(arr_all, 1)
        #     # and then repeat
        #     arr_all = dask.array.repeat(arr_all, self._num_timesteps, axis=1)
        da = xr.DataArray(arr_all, dims=dims, coords=coords)
        
        # stack the [x, y] dimensions into a `grid_index` dimension
        da = self.stack_grid_coords(da)
        
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
            dim="forcing_feature",
        )
        da_datetime_forcing = (da_datetime_forcing + 1) / 2  # Rescale to [0,1]
        da_datetime_forcing["forcing_feature"] = ["sin_hour", "cos_hour", "sin_year", "cos_year"]
        
        return da_datetime_forcing

    def get_vars_units(self, category: str) -> torch.List[str]:
        if category == "state":
            return self.config["dataset"]["var_units"]
        else:
            raise NotImplementedError(f"Category {category} not supported")
    
    def get_vars_names(self, category: str) -> torch.List[str]:
        if category == "state":
            return self.config["dataset"]["var_names"]
        elif category == "forcing":
            # XXX: this really shouldn't be hard-coded here, this should be in the config
            return ["toa_downwelling_shortwave_flux", "column_water", "sin_hour", "cos_hour", "sin_year", "cos_year"]
        elif category == "static":
            return ["surface_geopotential", "border_mask", "x", "y"]
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
        return self.config.coords_projection
    
    @property
    def grid_shape_state(self):
        return self.config.grid_shape_state
    
    @property
    def boundary_mask(self):
        xs, ys = self.get_xy(category="state", stacked=False)
        assert np.all(xs[0,:] == xs[-1,:])
        assert np.all(ys[:,0] == ys[:,-1])
        x = xs[0,:]
        y = ys[:,0]
        values = np.load(self.root_path / "static" / "border_mask.npy")
        da_mask = xr.DataArray(values, dims=["y", "x"], coords=dict(x=x, y=y), name="boundary_mask")
        da_mask_stacked_xy = self.stack_grid_coords(da_mask)
        return da_mask_stacked_xy


    def get_normalization_dataarray(self, category: str) -> xr.Dataset:
        """
        Return the normalization dataarray for the given category. This should contain
        a `{category}_mean` and `{category}_std` variable for each variable in the category.
        For `category=="state"`, the dataarray should also contain a `state_diff_mean` and
        `state_diff_std` variable for the one-step differences of the state variables.
        
        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The normalization dataarray for the given category, with variables for the mean
            and standard deviation of the variables (and differences for state variables).
        """
        def load_pickled_tensor(fn):
            return torch.load(self.root_path / "static" / fn).numpy()
            
        mean_diff_values = None
        std_diff_values = None
        if category == "state":
            mean_values = load_pickled_tensor("parameter_mean.pt")
            std_values = load_pickled_tensor("parameter_std.pt")
            mean_diff_values = load_pickled_tensor("diff_mean.pt")
            std_diff_values = load_pickled_tensor("diff_std.pt")
        elif category == "forcing":
            flux_stats = load_pickled_tensor("flux_stats.pt")  # (2,)
            flux_mean, flux_std = flux_stats
            # manually add hour sin/cos and day-of-year sin/cos stats for now
            # the mean/std for column_water is hardcoded for now
            mean_values = np.array([flux_mean, 0.34033957, 0.0, 0.0, 0.0, 0.0])
            std_values = np.array([flux_std, 0.4661307, 1.0, 1.0, 1.0, 1.0])

        else:
            raise NotImplementedError(f"Category {category} not supported")
        
        feature_dim_name = f"{category}_feature"
        variables = {
                f"{category}_mean": (feature_dim_name, mean_values),
                f"{category}_std": (feature_dim_name, std_values),
        }
        
        if mean_diff_values is not None and std_diff_values is not None:
            variables["state_diff_mean"] = (feature_dim_name, mean_diff_values)
            variables["state_diff_std"] = (feature_dim_name, std_diff_values)
        
        ds_norm = xr.Dataset(
            variables,
            coords={ feature_dim_name: self.get_vars_names(category=category) }
        )
        
        return ds_norm