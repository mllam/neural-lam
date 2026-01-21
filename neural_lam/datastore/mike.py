# Standard library
import collections
import copy
import functools
import warnings
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union

# Third-party
import cartopy.crs as ccrs
import geopandas as gpd
import mllam_data_prep as mdp
import numpy as np
import xarray as xr
from loguru import logger
from shapely import Point
import mikeio

# Local
from ..datastore.base import BaseDatastore

class MIKEDatastore(BaseDatastore):
    """
    Base class for weather data used in the neural-lam package. A datastore
    defines the interface for accessing weather data by providing methods to
    access the data in a processed format that can be used for training and
    evaluation of neural networks.

    NOTE: All methods return either primitive types, `numpy.ndarray`,
    `xarray.DataArray` or `xarray.Dataset` objects, not `pytorch.Tensor`
    objects. Conversion to `pytorch.Tensor` objects should be done in the
    `weather_dataset.WeatherDataset` class (which inherits from
    `torch.utils.data.Dataset` and uses the datastore to access the data).

    # Forecast vs analysis data
    If the datastore is used to represent forecast rather than analysis data,
    then the `is_forecast` attribute should be set to True, and returned data
    from `get_dataarray` is assumed to have `analysis_time` and `forecast_time`
    dimensions (rather than just `time`).

    # Ensemble vs deterministic data
    If the datastore is used to represent ensemble data, then the `is_ensemble`
    attribute should be set to True, and returned data from `get_dataarray` is
    assumed to have an `ensemble_member` dimension.

    # Grid index
    All methods that return data specific to a grid point (like
    `get_dataarray`) should have a single dimension named `grid_index` that
    represents the spatial grid index of the data. The actual x, y coordinates
    of the grid points should be stored in the `x` and `y` coordinates of the
    dataarray or dataset with the `grid_index` dimension as the coordinate for
    each of the `x` and `y` coordinates.
    """

    is_ensemble: bool = False
    is_forecast: bool = False

    CARTESIAN_COORDS = ["x", "y"]

    SHORT_NAME = "mike"

    def __init__(
        self,
        config_path: Path,
        reuse_existing: bool = True,
        preload_to_memory: bool = False,
        overload_stats_path: str | None = None,
    ):

        self._config_path = Path(config_path)
        self._config = mdp.Config.from_yaml_file(self._config_path)

        # Check for preload_to_memory in extra section
        preload_to_memory = self._config.extra.get(
            "preload_to_memory", preload_to_memory
        )

        # output path
        self._root_path = (
            Path(self._config.output.root_path)
            if self._config.output.root_path
            else self._config_path.parent
        )

        fp_ds = self._root_path / self._config_path.name.replace(
            ".yaml", ".zarr"
        )

        self._ds = None
        if reuse_existing and fp_ds.exists():
            # check that the zarr directory is newer than the config file
            if fp_ds.stat().st_mtime < self._config_path.stat().st_mtime:
                logger.warning(
                    "Config file has been modified since zarr was created. "
                    f"The old zarr archive (in {fp_ds}) will be used."
                    "To generate new zarr-archive, move the old one first."
                )
            self._ds = xr.open_zarr(fp_ds, consolidated=True)

        if self._ds is None:
            self._ds = mdp.create_dataset(config=self._config)
            self._ds.to_zarr(fp_ds)

        # Pre-load entire dataset into memory to eliminate disk I/O during
        # training for small datasets
        if preload_to_memory:
            print("Pre-loading dataset into memory...")
            self._ds = self._ds.load()
            print("Dataset loaded into memory.")

        print("The loaded datastore contains the following features:")
        for category in ["state", "forcing", "static"]:
            if len(self.get_vars_names(category)) > 0:
                var_names = self.get_vars_names(category)
                print(f" {category:<8s}: {' '.join(var_names)}")

        # check that all three train/val/test splits are available
        required_splits = ["train", "val", "test"]
        available_splits = list(self._ds.splits.split_name.values)
        if not all(split in available_splits for split in required_splits):
            raise ValueError(
                f"Missing required splits: {required_splits} in available "
                f"splits: {available_splits}"
            )

        print("With the following splits (over time):")
        for split in required_splits:
            da_split = self._ds.splits.sel(split_name=split)
            da_split_start = da_split.sel(split_part="start").load().item()
            da_split_end = da_split.sel(split_part="end").load().item()
            print(f" {split:<8s}: {da_split_start} to {da_split_end}")

        self.stats_datastore = None

        # check for path for mike_dataset in datastore (only interior) which is needed for plotting
        has_state = "state_fields" in self._config.inputs # currently only exists for interior datastore

        mike_dataset_cfg = self._config.extra.get("mike_dataset")
        if has_state:
            if not mike_dataset_cfg:
                raise ValueError("extra.mike_dataset must be set for interior (state) datastore.")
            if not Path(mike_dataset_cfg).exists():
                raise FileNotFoundError(f"Configured mike_dataset not found at {Path(mike_dataset_cfg)}")
            self.mike_dataset = mikeio.open(Path(mike_dataset_cfg))


    @property
    def root_path(self) -> Path:
        """
        The root path to the datastore. It is relative to this that any derived
        files (for example the graph components) are stored.

        Returns
        -------
        pathlib.Path
            The root path to the datastore.

        """
        return self._config_path.parent

    @property
    def config(self) -> collections.abc.Mapping:
        """The configuration of the datastore.

        Returns
        -------
        collections.abc.Mapping
            The configuration of the datastore, any dict like object can be
            returned.

        """
        return self._config

    @property
    def step_length(self) -> int:
        """The step length of the dataset in hours.

        Returns:
            int: The step length in hours.

        """
        da_dt = self._ds["time"].diff("time")
        total_sec = da_dt.dt.total_seconds().isel(time=0).astype(int)
        return (total_sec // 3600).item()

    @property
    def num_grid_points(self) -> int:
        """Return the number of grid points in the dataset.

        Returns
        -------
        int
            The number of grid points in the dataset.

        """
        return len(self._ds.grid_index)

    def get_vars_units(self, category: str) -> List[str]:
        """Get the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The units of the variables.

        """
        if category not in self._ds:
            warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature_units"].values.tolist()

    def get_vars_names(self, category: str) -> List[str]:
        """Get the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The names of the variables.

        """
        if category not in self._ds:
            warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature"].values.tolist()

    def get_vars_long_names(self, category: str) -> List[str]:
        """Get the long names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The long names of the variables.

        """
        if category not in self._ds:
            warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature"].values.tolist()

    def get_num_data_vars(self, category: str) -> int:
        """Get the number of data variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        int
            The number of data variables.

        """
        return len(self.get_vars_names(category))

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        """
        Return the standardization dataarray for the given category. This
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
        if self.stats_datastore is not None:
            # Get stats from self.stats_datastore instead
            return self.stats_datastore.get_standardization_dataarray(
                category=category
            )

        ops = ["mean", "std"]
        split = "train"
        stats_variables = {
            f"{category}__{split}__{op}": f"{category}_{op}" for op in ops
        }
        if category == "state":
            stats_variables.update(
                {f"state__{split}__diff_{op}": f"state_diff_{op}" for op in ops}
            )

        ds_stats = self._ds[stats_variables.keys()].rename(stats_variables)
        if "grid_index" in ds_stats.coords:
            ds_stats = ds_stats.isel(grid_index=0)
        return ds_stats

    def _standardize_datarray(
        self, da: xr.DataArray, category: str
    ) -> xr.DataArray:
        """
        Helper function to standardize a dataarray before returning it.

        Parameters
        ----------
        da: xr.DataArray
            The dataarray to standardize
        category : str
            The category of the dataarray (state/forcing/static), to load
            standardization statistics for.

        Returns
        -------
        xr.Dataarray
            The standardized dataarray
        """

        standard_da = self.get_standardization_dataarray(category=category)

        mean = standard_da[f"{category}_mean"]
        std = standard_da[f"{category}_std"]

        return (da - mean) / std

    def get_dataarray(
        self,
        category: str,
        split: Optional[str],
        standardize: bool = False,
    ) -> Union[xr.DataArray, None]:
        """
        Return the processed data (as a single `xr.DataArray`) for the given
        category of data and test/train/val-split that covers all the data (in
        space and time) of a given category (state/forcing/static). A
        datastore must be able to return for the "state" category, but
        "forcing" and "static" are optional (in which case the method should
        return `None`). For the "static" category the `split` is allowed to be
        `None` because the static data is the same for all splits.

        The returned dataarray is expected to at minimum have dimensions of
        `(grid_index, {category}_feature)` so that any spatial dimensions have
        been stacked into a single dimension and all variables and levels have
        been stacked into a single feature dimension named by the `category` of
        data being loaded.

        For categories of data that have a time dimension (i.e. not static
        data), the dataarray is expected additionally have `(analysis_time,
        elapsed_forecast_duration)` dimensions if `is_forecast` is True, or
        `(time)` if `is_forecast` is False.

        If the data is ensemble data, the dataarray is expected to have an
        additional `ensemble_member` dimension.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        split : str
            The time split to filter the dataset (train/val/test).
        standardize: bool
            If the dataarray should be returned standardized

        Returns
        -------
        xr.DataArray or None
            The xarray DataArray object with processed dataset.

        """
        if category not in self._ds and category == "forcing":
            warnings.warn("no forcing data found in datastore")
            return None

        da_category = self._ds[category]

        # set units on x y coordinates if missing
        for coord in ["x", "y"]:
            if "units" not in da_category[coord].attrs:
                da_category[coord].attrs["units"] = "m"

        # set multi-index for grid-index
        da_category = da_category.set_index(grid_index=self.CARTESIAN_COORDS)

        if "time" in da_category.dims:
            t_start = (
                self._ds.splits.sel(split_name=split)
                .sel(split_part="start")
                .load()
                .item()
            )
            t_end = (
                self._ds.splits.sel(split_name=split)
                .sel(split_part="end")
                .load()
                .item()
            )
            da_category = da_category.sel(time=slice(t_start, t_end))

        dim_order = self.expected_dim_order(category=category)
        da_category = da_category.transpose(*dim_order)

        if standardize:
            return self._standardize_datarray(da_category, category=category)

        return da_category

    @cached_property
    def boundary_mask(self) -> xr.DataArray:
        """
        Return the boundary mask for the dataset, with spatial dimensions
        stacked. Where the value is 1, the grid point is a boundary point, and
        where the value is 0, the grid point is not a boundary point.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimension ('grid_index',).
        """

        polygon_config = self._config.extra["boundary"]

        if polygon_config["method"] == "polygon":
            path = polygon_config["kwargs"]["polygon_path"]
            boundary_polygon = gpd.read_file(path).geometry.iloc[0]
            xy = self.get_xy(category="state", stacked=True)
            mask = np.array(
                [boundary_polygon.contains(Point(x, y)) for x, y in xy],
                dtype=int,
            )
        else:
            raise ValueError(
                f"Boundary method {polygon_config['method']} not implemented"
            )

        return xr.DataArray(
            data=mask, dims=("grid_index",), name="boundary_mask"
        )

    def get_xy(self, category: str, stacked: bool = True) -> np.ndarray:
        """
        Return the x, y coordinates of the dataset as a numpy arrays for a
        given category of data.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates. `stacked=False` is only
            meaningful for grid points on a regular-grid.

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset with shape `[n_grid_points, 2]`.
        """

        if not stacked:
            ValueError("An unstructured mesh needs stacked x and y coordinates")

        else:
            return np.stack(
                [self._ds[category].x.values, self._ds[category].y.values],
                axis=-1,
            )

    @property
    def coords_projection(self) -> ccrs.Projection:
        """
        Return the projection of the coordinates.

        NOTE: currently this expects the projection information to be in the
        `extra` section of the configuration file, with a `projection` key
        containing a `class_name` and `kwargs` for constructing the
        `cartopy.crs.Projection` object. This is a temporary solution until
        the projection information can be parsed in the produced dataset
        itself. `mllam-data-prep` ignores the contents of the `extra` section
        of the config file which is why we need to check that the necessary
        parts are there.

        Returns
        -------
        ccrs.Projection
            The projection of the coordinates.

        """
        if "projection" not in self._config.extra:
            raise ValueError(
                "projection information not found in the configuration file "
                f"({self._config_path}). Please add the projection information"
                "to the `extra` section of the config, by adding a "
                "`projection` key with the class name and kwargs of the "
                "projection."
            )

        projection_info = self._config.extra["projection"]
        if "class_name" not in projection_info:
            raise ValueError(
                "class_name not found in the projection information. Please "
                "add the class name of the projection to the `projection` key "
                "in the `extra` section of the config."
            )
        if "kwargs" not in projection_info:
            raise ValueError(
                "kwargs not found in the projection information. Please add "
                "the keyword arguments of the projection to the `projection` "
                "key in the `extra` section of the config."
            )

        class_name = projection_info["class_name"]
        ProjectionClass = getattr(ccrs, class_name)
        # need to copy otherwise we modify the dict stored in the dataclass
        # in-place
        kwargs = copy.deepcopy(projection_info["kwargs"])

        globe_kwargs = kwargs.pop("globe", {})
        if len(globe_kwargs) > 0:
            kwargs["globe"] = ccrs.Globe(**globe_kwargs)

        return ProjectionClass(**kwargs)

    @functools.lru_cache
    def get_xy_extent(self, category: str) -> List[float]:
        """
        Return the extent of the x, y coordinates for a given category of data.
        The extent should be returned as a list of 4 floats with `[xmin, xmax,
        ymin, ymax]` which can then be used to set the extent of a plot.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[float]
            The extent of the x, y coordinates.

        """
        xy = self.get_xy(category, stacked=True)
        extent = [
            xy[:, 0].min(),
            xy[:, 0].max(),
            xy[:, 1].min(),
            xy[:, 1].max(),
        ]
        return [float(v) for v in extent]

    @cached_property
    def state_feature_weights_values(self) -> List[float]:
        """
        Return the weights for each state feature as a list of floats. The
        weights are defined by the user in a config file for the datastore.

        Implementations of this method must assert that there is one weight for
        each state feature in the datastore. The weights can be used to scale
        the loss function for each state variable (e.g. via the standard
        deviation of the 1-step differences of the state variables).

        Returns:
            List[float]: The weights for each state feature.
        """
        raise NotImplementedError("Currently not implemented")

    @functools.lru_cache
    def expected_dim_order(
        self, category: Optional[str] = None
    ) -> tuple[str, ...]:
        """
        Return the expected dimension order for the dataarray or dataset
        returned by `get_dataarray` for the given category of data. The
        dimension order is the order of the dimensions in the dataarray or
        dataset, and is used to check that the data is in the expected format.

        This is necessary so that when stacking and unstacking the spatial grid
        we can ensure that the dimension order is the same as what is returned
        from `get_dataarray`. And also ensures that downstream uses of a
        datastore (e.g. WeatherDataset) sees the data in a common structure.

        If the category is None, then the it assumed that data only represents
        a 1D scalar field varying with grid-index.

        The order is constructed to match the order in `pytorch.Tensor` objects
        that will be constructed from the data so that the last two dimensions
        are always the grid-index and feature dimensions (i.e. the order is
        `[..., grid_index, {category}_feature]`), with any time-related and
        ensemble-number dimension(s) coming before these two.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[str]
            The expected dimension order for the dataarray or dataset.

        """
        dim_order = []

        if category is not None:
            if category != "static":
                # static data does not vary in time
                if self.is_forecast:
                    dim_order.extend(
                        ["analysis_time", "elapsed_forecast_duration"]
                    )
                elif not self.is_forecast:
                    dim_order.append("time")

            if self.is_ensemble and category == "state":
                # XXX: for now we only assume ensemble data for state variables
                dim_order.append("ensemble_member")

        dim_order.append("grid_index")

        if category is not None:
            dim_order.append(f"{category}_feature")

        return tuple(dim_order)
