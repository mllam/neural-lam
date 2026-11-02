"""Datastore implementation wrapping ``mllam-data-prep`` outputs."""

# Standard library
import copy
import functools
import warnings
from datetime import timedelta
from functools import cached_property
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import mllam_data_prep as mdp
import numpy as np
import xarray as xr
from loguru import logger
from numpy import ndarray

# Local
from ..utils.logging import log_on_rank_zero
from .base import BaseDatastore, CartesianGridShape


class MDPDatastore(BaseDatastore):
    """
    Datastore class for datasets made with the mllam_data_prep library
    (https://github.com/mllam/mllam-data-prep). This class wraps the
    `mllam_data_prep` library to do the necessary transforms to create the
    different categories (state/forcing/static) of data, with the actual
    transform to do being specified in the configuration file.
    """

    SHORT_NAME = "mdp"
    _ds: xr.Dataset

    def __init__(
        self,
        config_path: str | Path,
        n_boundary_points: int = 30,
        reuse_existing: bool = True,
    ) -> None:
        """
        Construct a new MDPDatastore from the configuration file at
        `config_path`. A boundary mask is created with `n_boundary_points`
        boundary points. If `reuse_existing` is True, the dataset is loaded
        from a zarr file if it exists (unless the config has been modified
        since the zarr was created), otherwise it is created from the
        configuration file.

        Parameters
        ----------
        config_path : str or Path
            The path to the configuration file, this will be fed to the
            `mllam_data_prep.Config.from_yaml_file` method to then call
            `mllam_data_prep.create_dataset` to create the dataset.
        n_boundary_points : int
            Width, in grid points, of the boundary along each x/y edge of the
            domain. 0 gives no boundary points.
        reuse_existing : bool
            Whether to reuse an existing dataset zarr file if it exists and its
            creation date is newer than the configuration file.

        Raises
        ------
        ValueError
            If the dataset does not contain all of the required
            train/val/test splits.

        """
        if n_boundary_points < 0:
            raise ValueError(
                f"n_boundary_points must be non-negative, got "
                f"{n_boundary_points}"
            )

        self._config_path = Path(config_path).resolve()
        self._root_path = self._config_path.parent
        self._config = mdp.Config.from_yaml_file(self._config_path)
        fp_ds = self._root_path / self._config_path.name.replace(
            ".yaml", ".zarr"
        )

        _ds = None
        if reuse_existing and fp_ds.exists():
            # check that the zarr directory is newer than the config file
            if fp_ds.stat().st_mtime < self._config_path.stat().st_mtime:
                logger.warning(
                    "Config file has been modified since zarr was created. "
                    f"The old zarr archive (in {fp_ds}) will be used."
                    "To generate new zarr-archive, move the old one first."
                )
            _ds = xr.open_zarr(fp_ds, consolidated=True)

        if _ds is None:
            _ds = mdp.create_dataset(config=self._config_for_creation())
            _ds.to_zarr(fp_ds)

        if "state" not in _ds and "forcing" not in _ds:
            raise ValueError(
                f"Datastore at {self._config_path} contains neither 'state' "
                "nor 'forcing' data. At least one is required."
            )

        self._ds = _ds
        self._n_boundary_points = n_boundary_points
        self.is_ensemble = (
            "state" in self._ds and "ensemble_member" in self._ds["state"].dims
        )
        self.has_ensemble_forcing = (
            "forcing" in self._ds
            and "ensemble_member" in self._ds["forcing"].dims
        )

        log_on_rank_zero(
            "The loaded datastore contains the following features:"
        )
        for category in ["state", "forcing", "static"]:
            if len(self.get_vars_names(category)) > 0:
                var_names = self.get_vars_names(category)
                log_on_rank_zero(f" {category:<8s}: {' '.join(var_names)}")

        # check that all three train/val/test splits are available
        required_splits = ["train", "val", "test"]
        available_splits = list(self._ds.splits.split_name.values)
        if not all(split in available_splits for split in required_splits):
            raise ValueError(
                f"Missing required splits: {required_splits} in available "
                f"splits: {available_splits}"
            )

        log_on_rank_zero("With the following splits (over time):")
        for split in required_splits:
            da_split = self._ds.splits.sel(split_name=split)
            da_split_start = da_split.sel(split_part="start").load().item()
            da_split_end = da_split.sel(split_part="end").load().item()
            log_on_rank_zero(
                f" {split:<8s}: {da_split_start} to {da_split_end}"
            )

        # find out the dimension order for the stacking to grid-index
        dim_order = None
        for input_dataset in self._config.inputs.values():
            dim_order_ = input_dataset.dim_mapping["grid_index"].dims
            if dim_order is None:
                dim_order = dim_order_
            else:
                assert dim_order == dim_order_, (
                    "all inputs must have the same dimension order"
                )

        if dim_order is None:
            raise ValueError("Could not determine dim_order from inputs.")
        self.spatial_coordinates = dim_order

    def _config_for_creation(self) -> mdp.Config:
        """Return the config to build the dataset from, with a relative
        `interior_dataset_config_path` made relative to this config file
        rather than the working directory.

        Returns
        -------
        mdp.Config
            The config, with any cropping path made absolute.
        """
        config = copy.deepcopy(self._config)
        domain_cropping = config.output.domain_cropping
        if domain_cropping is not None:
            interior_path = Path(domain_cropping.interior_dataset_config_path)
            if not interior_path.is_absolute():
                domain_cropping.interior_dataset_config_path = str(
                    self._root_path / interior_path
                )
        return config

    @property
    def root_path(self) -> Path:
        """The root path of the dataset.

        Returns
        -------
        Path
            The root path of the dataset.

        """
        return self._root_path

    @property
    def config(self) -> mdp.Config:
        """The configuration of the dataset.

        Returns
        -------
        mdp.Config
            The configuration of the dataset.

        """
        return self._config

    @property
    def step_length(self) -> timedelta:
        """The length of the time steps as a time interval.

        Returns
        -------
        timedelta
            The length of the time steps as a datetime.timedelta object.

        """
        da_dt = self._ds["time"].diff("time")
        total_sec = da_dt.dt.total_seconds().isel(time=0).astype(int)
        return timedelta(seconds=int(total_sec.item()))

    def get_vars_units(self, category: str) -> list[str]:
        """Return the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        list[str]
            The units of the variables in the given category.

        """
        if category not in self._ds:
            if category in ("forcing", "static"):
                warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature_units"].values.tolist()

    def get_vars_names(self, category: str) -> list[str]:
        """Return the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        list[str]
            The names of the variables in the given category.

        """
        if category not in self._ds:
            if category in ("forcing", "static"):
                warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature"].values.tolist()

    def get_vars_long_names(self, category: str) -> list[str]:
        """
        Return the long names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        list[str]
            The long names of the variables in the given category.

        """
        if category not in self._ds:
            if category in ("forcing", "static"):
                warnings.warn(f"no {category} data found in datastore")
            return []
        return self._ds[f"{category}_feature_long_name"].values.tolist()

    def get_num_data_vars(self, category: str) -> int:
        """Return the number of variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        int
            The number of variables in the given category.

        """
        return len(self.get_vars_names(category))

    def get_dataarray(
        self,
        category: str,
        split: str | None,
        standardize: bool = False,
    ) -> xr.DataArray | None:
        """
        Return the processed data (as a single `xr.DataArray`) for the given
        category of data and test/train/val-split that covers all the data (in
        space and time) of a given category (state/forcing/static). "state" is
        the only required category, for other categories, the method will
        return `None` if the category is not found in the datastore.

        The returned dataarray will at minimum have dimensions of `(grid_index,
        {category}_feature)` so that any spatial dimensions have been stacked
        into a single dimension and all variables and levels have been stacked
        into a single feature dimension named by the `category` of data being
        loaded.

        For categories of data that have a time dimension (i.e. not static
        data), the dataarray will additionally have `(analysis_time,
        elapsed_forecast_duration)` dimensions if `is_forecast` is True, or
        `(time)` if `is_forecast` is False.

        If we have multiple ensemble members of state data, the returned state
        dataarray will have an additional `ensemble_member` dimension. If
        `has_ensemble_forcing=True`, the returned forcing dataarray will also
        have an additional `ensemble_member` dimension.

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
        if category not in self._ds and category in ("forcing", "static"):
            warnings.warn(f"no {category} data found in datastore")
            return None

        da_category = self._ds[category]

        # set units on x y coordinates if missing
        geographic_units = {
            "longitude": "degrees_east",
            "latitude": "degrees_north",
            "lon": "degrees_east",
            "lat": "degrees_north",
        }
        for coord in self.spatial_coordinates:
            if "units" not in da_category[coord].attrs:
                da_category[coord].attrs["units"] = geographic_units.get(
                    coord, "m"
                )

        # set multi-index for grid-index
        da_category = da_category.set_index(grid_index=self.spatial_coordinates)

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

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        """
        Return the standardization dataarray for the given category. This
        should contain a `{category}_mean` and `{category}_std` variable for
        each variable in the category.
        For `category=="state"`, the dataarray should also contain a
        `state_diff_mean_standardized` and `state_diff_std_standardized`
        variable for the one-step differences of the state variables.

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
        ops = ["mean", "std"]
        split = "train"
        stats_variables = {
            f"{category}__{split}__{op}": f"{category}_{op}" for op in ops
        }

        ds_stats = self._ds[stats_variables.keys()].rename(stats_variables)

        # Add standardized state diff stats
        if category == "state":
            ds_stats = ds_stats.assign(
                {
                    f"state_diff_{op}_standardized": self._ds[
                        f"state__{split}__diff_{op}"
                    ]
                    / ds_stats["state_std"]
                    for op in ops
                }
            )

        return ds_stats

    @cached_property
    def boundary_mask(self) -> xr.DataArray:
        """
        Produce a 0/1 mask for the boundary points of the dataset, these will
        sit at the edges of the domain (in x/y extent) and will be used to mask
        out the boundary points from the loss function and to overwrite the
        boundary points from the prediction. For now this is created when the
        mask is requested, but in the future this could be saved to the zarr
        file.

        Returns
        -------
        xr.DataArray
            A 0/1 mask for the boundary points of the dataset, where 1 is a
            boundary point and 0 is not.

        Raises
        ------
        NotImplementedError
            If the datastore provides no ``state`` data.

        """
        if "state" not in self._ds:
            raise NotImplementedError(
                "`boundary_mask` is only defined for datastores with `state` "
                "data."
            )

        ds_unstacked = self.unstack_grid_coords(da_or_ds=self._ds)
        da_state_variable = (
            ds_unstacked["state"].isel(time=0).isel(state_feature=0)
        )
        da_domain_allzero = xr.zeros_like(da_state_variable)
        # slice(n, -n) is empty (not "the whole array") when n == 0
        n = self._n_boundary_points
        interior_slice = slice(n, -n) if n > 0 else slice(None)
        ds_unstacked["boundary_mask"] = da_domain_allzero.isel(
            x=interior_slice,
            y=interior_slice,
        )
        ds_unstacked["boundary_mask"] = ds_unstacked.boundary_mask.fillna(
            1
        ).astype(int)
        boundary_mask = self.stack_grid_coords(
            da_or_ds=ds_unstacked.boundary_mask
        )
        assert isinstance(boundary_mask, xr.DataArray)
        return boundary_mask

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

        Raises
        ------
        ValueError
            If the `projection` entry is missing from the `extra` section of
            the config, or if its `class_name` or `kwargs` keys are missing.

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

    @cached_property
    def _grid_xy_counts(self) -> tuple[int, int, int]:
        """Number of distinct ``x`` values, ``y`` values and ``(x, y)`` pairs,
        read from the coordinates since unstacking is gated on
        `is_on_rectilinear_grid`, which is derived from these.

        Returns
        -------
        tuple[int, int, int]
            Number of distinct ``x`` values, ``y`` values and ``(x, y)``
            pairs.
        """
        category = "state" if "state" in self._ds else "forcing"
        da_cat = self._ds[category]
        xdim, ydim = self.spatial_coordinates
        xs = da_cat[xdim].values
        ys = da_cat[ydim].values
        n_pairs = np.unique(np.stack([xs, ys], axis=1), axis=0).shape[0]
        return np.unique(xs).size, np.unique(ys).size, n_pairs

    @cached_property
    def grid_shape_state(self) -> CartesianGridShape:
        """The shape of the cartesian grid for the state variables.

        Returns
        -------
        CartesianGridShape
            The shape of the cartesian grid for the state variables.

        Raises
        ------
        NotImplementedError
            If the datastore is not on a rectilinear grid, e.g. after
            ``domain_cropping``. Use `num_grid_points` instead.
        """
        self._check_on_rectilinear_grid("grid_shape_state")
        n_x, n_y, _ = self._grid_xy_counts
        return CartesianGridShape(x=n_x, y=n_y)

    def get_xy(self, category: str, stacked: bool) -> ndarray:
        """
        Return the x, y coordinates of the dataset (i.e. the Cartesian
        coordinates of the regular gridded dataset).

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates. `stacked=False` requires
            `is_on_rectilinear_grid`.

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset, returned differently based on
            the value of `stacked`:

            * `stacked==True`: shape `(num_grid_nodes, 2)`, with
              num_grid_nodes = grid_size_x * grid_size_y on a rectilinear grid.
            * `stacked==False`: shape `(grid_size_x, grid_size_y, 2)`

        """
        xdim, ydim = self.spatial_coordinates

        if stacked and not self.is_on_rectilinear_grid:
            # There is no x/y grid to unstack onto, so return the
            # coordinates of the grid points in `grid_index` order
            da_cat = self._ds[category]
            return np.stack([da_cat[xdim].values, da_cat[ydim].values], axis=1)

        # assume variables are stored in dimensions [grid_index, ...]
        ds_category = self.unstack_grid_coords(da_or_ds=self._ds[category])

        da_xs = ds_category[xdim]
        da_ys = ds_category[ydim]

        assert da_xs.ndim == da_ys.ndim == 1, (
            f"{xdim} and {ydim} coordinates must be 1D"
        )

        da_x, da_y = xr.broadcast(da_xs, da_ys)
        da_xy = xr.concat([da_x, da_y], dim="grid_coord")

        if stacked:
            da_xy = da_xy.stack(grid_index=self.spatial_coordinates).transpose(
                "grid_index",
                "grid_coord",
            )
        else:
            dims = [
                xdim,
                ydim,
                "grid_coord",
            ]
            da_xy = da_xy.transpose(*dims)

        return da_xy.values

    @functools.lru_cache
    def get_lat_lon(self, category: str) -> np.ndarray:
        """
        Return the longitude, latitude coordinates of the dataset as numpy
        array for a given category of data.
        Override in MDP to use lat/lons directly from xr.Dataset, if available.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        np.ndarray
            The longitude, latitude coordinates of the dataset
            with shape `[num_grid_nodes, 2]`.
        """
        # Check first if lat/lon saved in ds
        lookup_ds = self._ds
        if "latitude" in lookup_ds.coords and "longitude" in lookup_ds.coords:
            lon = lookup_ds.longitude
            lat = lookup_ds.latitude
        elif "lat" in lookup_ds.coords and "lon" in lookup_ds.coords:
            lon = lookup_ds.lon
            lat = lookup_ds.lat
        else:
            # Not saved, use method from BaseDatastore to derive from x/y
            return super().get_lat_lon(category)

        coords = np.stack((lon.values, lat.values), axis=1)
        return coords

    @property
    def is_on_rectilinear_grid(self) -> bool:
        """Whether ``grid_index`` can be unstacked into a complete 2D
        ``x``/``y`` rectilinear grid, i.e. every combination of the distinct
        ``x`` and ``y`` values is present exactly once. The spacing need not
        be uniform. A datastore cropped with ``domain_cropping`` is not.

        Returns
        -------
        bool
            ``True`` when every cell of the ``x``/``y`` grid is present
            exactly once.
        """
        n_x, n_y, n_pairs = self._grid_xy_counts
        return self.num_grid_points == n_pairs == n_x * n_y

    @property
    def num_grid_points(self) -> int:
        """Return the number of grid points in the dataset.

        Returns
        -------
        int
            The number of grid points in the dataset.
        """
        return len(self._ds.grid_index)
