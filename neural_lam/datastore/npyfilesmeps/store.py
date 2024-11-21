"""
Numpy-files based datastore to support the MEPS example dataset introduced in
neural-lam v0.1.0.
"""

# Standard library
import functools
import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import List

# Third-party
import cartopy.crs as ccrs
import dask
import dask.array
import dask.delayed
import numpy as np
import parse
import torch
import xarray as xr
from xarray.core.dataarray import DataArray

# Local
from ..base import BaseRegularGridDatastore, CartesianGridShape
from .config import NpyDatastoreConfig

STATE_FILENAME_FORMAT = "nwp_{analysis_time:%Y%m%d%H}_mbr{member_id:03d}.npy"
TOA_SW_DOWN_FLUX_FILENAME_FORMAT = (
    "nwp_toa_downwelling_shortwave_flux_{analysis_time:%Y%m%d%H}.npy"
)
OPEN_WATER_FILENAME_FORMAT = "wtr_{analysis_time:%Y%m%d%H}.npy"


def _load_np(fp, add_feature_dim, feature_dim_mask=None):
    arr = np.load(fp)
    if add_feature_dim:
        arr = arr[..., np.newaxis]
    if feature_dim_mask is not None:
        arr = arr[..., feature_dim_mask]
    return arr


class NpyFilesDatastoreMEPS(BaseRegularGridDatastore):
    __doc__ = f"""
    Represents a dataset stored as numpy files on disk. The dataset is assumed
    to be stored in a directory structure where each sample is stored in a
    separate file. The file-name format is assumed to be
    '{STATE_FILENAME_FORMAT}'

    The MEPS dataset is organised into three splits: train, val, and test. Each
    split has a set of files which are:

    - `{STATE_FILENAME_FORMAT}`:
        The state variables for a forecast started at `analysis_time` with
        member id `member_id`. The dimensions of the array are
        `[forecast_timestep, y, x, feature]`.

    - `{TOA_SW_DOWN_FLUX_FILENAME_FORMAT}`:
        The top-of-atmosphere downwelling shortwave flux at `time`. The
        dimensions of the array are `[forecast_timestep, y, x]`.

    - `{OPEN_WATER_FILENAME_FORMAT}`:
        The open water fraction at `time`. The dimensions of the array are
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
    SHORT_NAME = "npyfilesmeps"

    is_ensemble = True
    is_forecast = True

    def __init__(
        self,
        config_path,
    ):
        """
        Create a new NpyFilesDatastore using the configuration file at the
        given path. The config file should be a YAML file and will be loaded
        into an instance of the `NpyDatastoreConfig` dataclass.

        Internally, the datastore uses dask.delayed to load the data from the
        numpy files, so that the data isn't actually loaded until it's needed.

        Parameters
        ----------
        config_path : str
            The path to the configuration file for the datastore.

        """
        self._config_path = Path(config_path)
        self._root_path = self._config_path.parent
        self._config = NpyDatastoreConfig.from_yaml_file(self._config_path)

        self._num_ensemble_members = self.config.dataset.num_ensemble_members
        self._num_timesteps = self.config.dataset.num_timesteps
        self._step_length = self.config.dataset.step_length
        self._remove_state_features_with_index = (
            self.config.dataset.remove_state_features_with_index
        )

    @property
    def root_path(self) -> Path:
        """
        The root path of the datastore on disk. This is the directory relative
        to which graphs and other files can be stored.

        Returns
        -------
        Path
            The root path of the datastore

        """
        return self._root_path

    @property
    def config(self) -> NpyDatastoreConfig:
        """The configuration for the datastore.

        Returns
        -------
        NpyDatastoreConfig
            The configuration for the datastore.

        """
        return self._config

    def get_dataarray(self, category: str, split: str) -> DataArray:
        """
        Get the data array for the given category and split of data. If the
        category is 'state', the data array will be a concatenation of the data
        arrays for all ensemble members. The data will be loaded as a dask
        array, so that the data isn't actually loaded until it's needed.

        Parameters
        ----------
        category : str
            The category of the data to load. One of 'state', 'forcing', or
            'static'.
        split : str
            The dataset split to load the data for. One of 'train', 'val', or
            'test'.

        Returns
        -------
        xr.DataArray
            The data array for the given category and split, with dimensions
            per category:
            state:     `[elapsed_forecast_duration, analysis_time, grid_index,
                        feature, ensemble_member]`
            forcing:   `[elapsed_forecast_duration, analysis_time, grid_index,
                        feature]`
            static:    `[grid_index, feature]`

        """
        if category == "state":
            das = []
            # for the state category, we need to load all ensemble members
            for member in range(self._num_ensemble_members):
                da_member = self._get_single_timeseries_dataarray(
                    features=self.get_vars_names(category="state"),
                    split=split,
                    member=member,
                )
                das.append(da_member)
            da = xr.concat(das, dim="ensemble_member")

        elif category == "forcing":
            # the forcing features are in separate files, so we need to load
            # them separately
            features = ["toa_downwelling_shortwave_flux", "open_water_fraction"]
            das = [
                self._get_single_timeseries_dataarray(
                    features=[feature], split=split
                )
                for feature in features
            ]
            da = xr.concat(das, dim="feature")

            # add datetime forcing as a feature
            # to do this we create a forecast time variable which has the
            # dimensions of (analysis_time, elapsed_forecast_duration) with
            # values that are the actual forecast time of each time step. By
            # calling .chunk({"elapsed_forecast_duration": 1}) this time
            # variable is turned into a dask array and so execution of the
            # calculation is delayed until the feature values are actually
            # used.
            da_forecast_time = (
                da.analysis_time + da.elapsed_forecast_duration
            ).chunk({"elapsed_forecast_duration": 1})
            da_datetime_forcing_features = self._calc_datetime_forcing_features(
                da_time=da_forecast_time
            )
            da = xr.concat([da, da_datetime_forcing_features], dim="feature")

        elif category == "static":
            # the static features are collected in three files:
            # - surface_geopotential
            # - border_mask
            # - x, y
            das = []
            for features in [
                ["surface_geopotential"],
                ["border_mask"],
                ["x", "y"],
            ]:
                da = self._get_single_timeseries_dataarray(
                    features=features, split=split
                )
                das.append(da)
            da = xr.concat(das, dim="feature")

        else:
            raise NotImplementedError(category)

        da = da.rename(dict(feature=f"{category}_feature"))

        # stack the [x, y] dimensions into a `grid_index` dimension
        da = self.stack_grid_coords(da)

        # check that we have the right features
        actual_features = da[f"{category}_feature"].values.tolist()
        expected_features = self.get_vars_names(category=category)
        if actual_features != expected_features:
            raise ValueError(
                f"Expected features {expected_features}, got {actual_features}"
            )

        dim_order = self.expected_dim_order(category=category)
        da = da.transpose(*dim_order)

        return da

    def _get_single_timeseries_dataarray(
        self, features: List[str], split: str, member: int = None
    ) -> DataArray:
        """
        Get the data array spanning the complete time series for a given set of
        features and split of data. For state features the `member` argument
        should be specified to select the ensemble member to load. The data
        will be loaded using dask.delayed, so that the data isn't actually
        loaded until it's needed.

        Parameters
        ----------
        features : List[str]
            The list of features to load the data for. For the 'state'
            category, this should be the result of
            `self.get_vars_names(category="state")`, for the 'forcing' category
            this should be the list of forcing features to load, and for the
            'static' category this should be the list of static features to
            load.
        split : str
            The dataset split to load the data for. One of 'train', 'val', or
            'test'.
        member : int, optional
            The ensemble member to load. Only applicable for the 'state'
            category.

        Returns
        -------
        xr.DataArray
            The data array for the given category and split, with dimensions
            `[elapsed_forecast_duration, analysis_time, grid_index, feature]`
            for all categories of data

        """
        if (
            set(features).difference(self.get_vars_names(category="static"))
            == set()
        ):
            assert split in (
                "train",
                "val",
                "test",
                None,
            ), "Unknown dataset split"
        else:
            assert split in (
                "train",
                "val",
                "test",
            ), f"Unknown dataset split {split} for features {features}"

        if member is not None and features != self.get_vars_names(
            category="state"
        ):
            raise ValueError(
                "Member can only be specified for the 'state' category"
            )

        concat_axis = 0

        file_params = {}
        add_feature_dim = False
        features_vary_with_analysis_time = True
        feature_dim_mask = None
        if features == self.get_vars_names(category="state"):
            filename_format = STATE_FILENAME_FORMAT
            file_dims = ["elapsed_forecast_duration", "y", "x", "feature"]
            # only select one member for now
            file_params["member_id"] = member
            fp_samples = self.root_path / "samples" / split
            if self._remove_state_features_with_index:
                n_to_drop = len(self._remove_state_features_with_index)
                feature_dim_mask = np.ones(
                    len(features) + n_to_drop, dtype=bool
                )
                feature_dim_mask[self._remove_state_features_with_index] = False
        elif features == ["toa_downwelling_shortwave_flux"]:
            filename_format = TOA_SW_DOWN_FLUX_FILENAME_FORMAT
            file_dims = ["elapsed_forecast_duration", "y", "x", "feature"]
            add_feature_dim = True
            fp_samples = self.root_path / "samples" / split
        elif features == ["open_water_fraction"]:
            filename_format = OPEN_WATER_FILENAME_FORMAT
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
            fp_samples = self.root_path / "samples" / split
        elif features == ["surface_geopotential"]:
            filename_format = "surface_geopotential.npy"
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
            features_vary_with_analysis_time = False
            # XXX: surface_geopotential is the same for all splits, and so
            # saved in static/
            fp_samples = self.root_path / "static"
        elif features == ["border_mask"]:
            filename_format = "border_mask.npy"
            file_dims = ["y", "x", "feature"]
            add_feature_dim = True
            features_vary_with_analysis_time = False
            # XXX: border_mask is the same for all splits, and so saved in
            # static/
            fp_samples = self.root_path / "static"
        elif features == ["x", "y"]:
            filename_format = "nwp_xy.npy"
            # NB: for x, y the feature dimension is the first one
            file_dims = ["feature", "y", "x"]
            features_vary_with_analysis_time = False
            # XXX: x, y are the same for all splits, and so saved in static/
            fp_samples = self.root_path / "static"
        else:
            raise NotImplementedError(
                f"Reading of variables set `{features}` not supported"
            )

        if features_vary_with_analysis_time:
            dims = ["analysis_time"] + file_dims
        else:
            dims = file_dims

        coords = {}
        arr_shape = []

        xy = self.get_xy(category="state", stacked=False)
        xs = xy[:, :, 0]
        ys = xy[:, :, 1]
        # Check if x-coordinates are constant along columns
        assert np.allclose(xs, xs[:, [0]]), "x-coordinates are not constant"
        # Check if y-coordinates are constant along rows
        assert np.allclose(ys, ys[[0], :]), "y-coordinates are not constant"
        # Extract unique x and y coordinates
        x = xs[:, 0]  # Unique x-coordinates (changes along the first axis)
        y = ys[0, :]  # Unique y-coordinates (changes along the second axis)
        for d in dims:
            if d == "elapsed_forecast_duration":
                coord_values = (
                    self.step_length
                    * np.arange(self._num_timesteps)
                    * np.timedelta64(1, "h")
                )
            elif d == "analysis_time":
                coord_values = self._get_analysis_times(split=split)
            elif d == "y":
                coord_values = y
            elif d == "x":
                coord_values = x
            elif d == "feature":
                coord_values = features
            else:
                raise NotImplementedError(f"Dimension {d} not supported")

            coords[d] = coord_values
            if d != "analysis_time":
                # analysis_time varies across the different files, but not
                # within a single file
                arr_shape.append(len(coord_values))

        if features_vary_with_analysis_time:
            filepaths = [
                fp_samples
                / filename_format.format(
                    analysis_time=analysis_time, **file_params
                )
                for analysis_time in coords["analysis_time"]
            ]
        else:
            filepaths = [fp_samples / filename_format.format(**file_params)]

        # use dask.delayed to load the numpy files, so that loading isn't
        # done until the data is actually needed
        arrays = [
            dask.array.from_delayed(
                dask.delayed(_load_np)(
                    fp=fp,
                    add_feature_dim=add_feature_dim,
                    feature_dim_mask=feature_dim_mask,
                ),
                shape=arr_shape,
                dtype=np.float32,
            )
            for fp in filepaths
        ]

        # read a single timestep and check the shape
        arr0 = arrays[0].compute()
        if not list(arr0.shape) == arr_shape:
            raise Exception(
                f"Expected shape {arr_shape} for a single file, got "
                f"{list(arr0.shape)}. Maybe the number of features given "
                f"in the datastore config ({features}) is incorrect?"
            )

        if features_vary_with_analysis_time:
            arr_all = dask.array.stack(arrays, axis=concat_axis)
        else:
            arr_all = arrays[0]

        da = xr.DataArray(arr_all, dims=dims, coords=coords)

        return da

    def _get_analysis_times(self, split) -> List[np.datetime64]:
        """Get the analysis times for the given split by parsing the filenames
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
        pattern = re.sub(r"{analysis_time:[^}]*}", "*", STATE_FILENAME_FORMAT)
        pattern = re.sub(r"{member_id:[^}]*}", "*", pattern)

        sample_dir = self.root_path / "samples" / split
        sample_files = sample_dir.glob(pattern)
        times = []
        for fp in sample_files:
            name_parts = parse.parse(STATE_FILENAME_FORMAT, fp.name)
            times.append(name_parts["analysis_time"])

        if len(times) == 0:
            raise ValueError(
                f"No files found in {sample_dir} with pattern {pattern}"
            )

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
        da_datetime_forcing["feature"] = [
            "sin_hour",
            "cos_hour",
            "sin_year",
            "cos_year",
        ]

        return da_datetime_forcing

    def get_vars_units(self, category: str) -> List[str]:
        if category == "state":
            return self.config.dataset.var_units
        elif category == "forcing":
            return [
                "W/m^2",
                "1",
                "1",
                "1",
                "1",
                "1",
            ]
        elif category == "static":
            return ["m^2/s^2", "1", "m", "m"]
        else:
            raise NotImplementedError(f"Category {category} not supported")

    def get_vars_names(self, category: str) -> List[str]:
        if category == "state":
            return self.config.dataset.var_names
        elif category == "forcing":
            # XXX: this really shouldn't be hard-coded here, this should be in
            # the config
            return [
                "toa_downwelling_shortwave_flux",
                "open_water_fraction",
                "sin_hour",
                "cos_hour",
                "sin_year",
                "cos_year",
            ]
        elif category == "static":
            return ["surface_geopotential", "border_mask", "x", "y"]
        else:
            raise NotImplementedError(f"Category {category} not supported")

    def get_vars_long_names(self, category: str) -> List[str]:
        if category == "state":
            return self.config.dataset.var_longnames
        else:
            # TODO: should we add these?
            return self.get_vars_names(category=category)

    def get_num_data_vars(self, category: str) -> int:
        return len(self.get_vars_names(category=category))

    def get_xy(self, category: str, stacked: bool) -> np.ndarray:
        """Return the x, y coordinates of the dataset.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates.

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset (with x first then y second),
            returned differently based on the value of `stacked`:
            - `stacked==True`: shape `(n_grid_points, 2)` where
                                      n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(N_x, N_y, 2)`

        """

        # the array on disk has shape [2, N_y, N_x], where dimension 0
        # contains the [x,y] coordinate pairs for each grid point
        arr = np.load(self.root_path / "static" / "nwp_xy.npy")
        arr_shape = arr.shape

        assert arr_shape[0] == 2, "Expected 2D array"
        grid_shape = self.grid_shape_state
        assert arr_shape[1:] == (grid_shape.y, grid_shape.x), "Unexpected shape"

        arr = arr.transpose(2, 1, 0)

        if stacked:
            return arr.reshape(-1, 2)
        else:
            return arr

    @property
    def step_length(self) -> int:
        """The length of each time step in hours.

        Returns
        -------
        int
            The length of each time step in hours.

        """
        return self._step_length

    @cached_property
    def grid_shape_state(self) -> CartesianGridShape:
        """The shape of the cartesian grid for the state variables.

        Returns
        -------
        CartesianGridShape
            The shape of the cartesian grid for the state variables.

        """
        ny, nx = self.config.grid_shape_state
        return CartesianGridShape(x=nx, y=ny)

    @cached_property
    def boundary_mask(self) -> xr.DataArray:
        """The boundary mask for the dataset. This is a binary mask that is 1
        where the grid cell is on the boundary of the domain, and 0 otherwise.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions `[grid_index]`.

        """
        xy = self.get_xy(category="state", stacked=False)
        xs = xy[:, :, 0]
        ys = xy[:, :, 1]
        # Check if x-coordinates are constant along columns
        assert np.allclose(xs, xs[:, [0]]), "x-coordinates are not constant"
        # Check if y-coordinates are constant along rows
        assert np.allclose(ys, ys[[0], :]), "y-coordinates are not constant"
        # Extract unique x and y coordinates
        x = xs[:, 0]  # Unique x-coordinates (changes along the first axis)
        y = ys[0, :]  # Unique y-coordinates (changes along the second axis)
        values = np.load(self.root_path / "static" / "border_mask.npy")
        da_mask = xr.DataArray(
            values, dims=["y", "x"], coords=dict(x=x, y=y), name="boundary_mask"
        )
        da_mask_stacked_xy = self.stack_grid_coords(da_mask).astype(int)
        return da_mask_stacked_xy

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        """Return the standardization dataarray for the given category. This
        should contain a `{category}_mean` and `{category}_std` variable for
        each variable in the category. For `category=="state"`, the dataarray
        should also contain a `state_diff_mean` and `state_diff_std` variable
        for the one- step differences of the state variables.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The standardization dataarray for the given category, with
            variables for the mean and standard deviation of the variables (and
            differences for state variables).

        """

        def load_pickled_tensor(fn):
            return torch.load(
                self.root_path / "static" / fn, weights_only=True
            ).numpy()

        mean_diff_values = None
        std_diff_values = None
        if category == "state":
            mean_values = load_pickled_tensor("parameter_mean.pt")
            std_values = load_pickled_tensor("parameter_std.pt")
            try:
                mean_diff_values = load_pickled_tensor("diff_mean.pt")
                std_diff_values = load_pickled_tensor("diff_std.pt")
            except FileNotFoundError:
                warnings.warn(f"Could not load diff mean/std for {category}")
                # XXX: this is a hack, but when running
                # compute_standardization_stats the diff mean/std files are
                # created, but require the std and mean files
                mean_diff_values = np.empty_like(mean_values)
                std_diff_values = np.empty_like(std_values)

        elif category == "forcing":
            flux_stats = load_pickled_tensor("flux_stats.pt")  # (2,)
            flux_mean, flux_std = flux_stats
            # manually add hour sin/cos and day-of-year sin/cos stats for now
            # the mean/std for open_water_fraction is hardcoded for now
            mean_values = np.array([flux_mean, 0.0, 0.0, 0.0, 0.0, 0.0])
            std_values = np.array([flux_std, 1.0, 1.0, 1.0, 1.0, 1.0])

        elif category == "static":
            da_static = self.get_dataarray(category="static", split="train")
            da_static_mean = da_static.mean(dim=["grid_index"]).compute()
            da_static_std = da_static.std(dim=["grid_index"]).compute()
            mean_values = da_static_mean.values
            std_values = da_static_std.values
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
            coords={feature_dim_name: self.get_vars_names(category=category)},
        )

        return ds_norm

    @functools.cached_property
    def coords_projection(self) -> ccrs.Projection:
        """The projection of the spatial coordinates.

        Returns
        -------
        ccrs.Projection
            The projection of the spatial coordinates.

        """
        proj_class_name = self.config.projection.class_name
        ProjectionClass = getattr(ccrs, proj_class_name)
        proj_params = self.config.projection.kwargs
        return ProjectionClass(**proj_params)
