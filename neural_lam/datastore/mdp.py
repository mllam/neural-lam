# Standard library
import copy
import warnings
from functools import cached_property
from pathlib import Path
from typing import List

# Third-party
import cartopy.crs as ccrs
import mllam_data_prep as mdp
import xarray as xr
from loguru import logger
from numpy import ndarray

# Local
from ..utils import rank_zero_print
from .base import BaseRegularGridDatastore, CartesianGridShape


class MDPDatastore(BaseRegularGridDatastore):
    """
    Datastore class for datasets made with the mllam_data_prep library
    (https://github.com/mllam/mllam-data-prep). This class wraps the
    `mllam_data_prep` library to do the necessary transforms to create the
    different categories (state/forcing/static) of data, with the actual
    transform to do being specified in the configuration file.
    """

    SHORT_NAME = "mdp"

    def __init__(self, config_path, n_boundary_points=30, reuse_existing=True):
        """
        Construct a new MDPDatastore from the configuration file at
        `config_path`. A boundary mask is created with `n_boundary_points`
        boundary points. If `reuse_existing` is True, the dataset is loaded
        from a zarr file if it exists (unless the config has been modified
        since the zarr was created), otherwise it is created from the
        configuration file.

        Parameters
        ----------
        config_path : str
            The path to the configuration file, this will be fed to the
            `mllam_data_prep.Config.from_yaml_file` method to then call
            `mllam_data_prep.create_dataset` to create the dataset.
        n_boundary_points : int
            The number of boundary points to use in the boundary mask.
        reuse_existing : bool
            Whether to reuse an existing dataset zarr file if it exists and its
            creation date is newer than the configuration file.

        """
        self._config_path = Path(config_path)
        self._root_path = self._config_path.parent
        self._config = mdp.Config.from_yaml_file(self._config_path)
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
        self._n_boundary_points = n_boundary_points

        rank_zero_print("The loaded datastore contains the following features:")
        for category in ["state", "forcing", "static"]:
            if len(self.get_vars_names(category)) > 0:
                var_names = self.get_vars_names(category)
                rank_zero_print(f" {category:<8s}: {' '.join(var_names)}")

        # check that all three train/val/test splits are available
        required_splits = ["train", "val", "test"]
        available_splits = list(self._ds.splits.split_name.values)
        if not all(split in available_splits for split in required_splits):
            raise ValueError(
                f"Missing required splits: {required_splits} in available "
                f"splits: {available_splits}"
            )

        rank_zero_print("With the following splits (over time):")
        for split in required_splits:
            da_split = self._ds.splits.sel(split_name=split)
            da_split_start = da_split.sel(split_part="start").load().item()
            da_split_end = da_split.sel(split_part="end").load().item()
            rank_zero_print(f" {split:<8s}: {da_split_start} to {da_split_end}")

        # find out the dimension order for the stacking to grid-index
        dim_order = None
        for input_dataset in self._config.inputs.values():
            dim_order_ = input_dataset.dim_mapping["grid_index"].dims
            if dim_order is None:
                dim_order = dim_order_
            else:
                assert (
                    dim_order == dim_order_
                ), "all inputs must have the same dimension order"

        self.CARTESIAN_COORDS = dim_order

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
    def step_length(self) -> int:
        """The length of the time steps in hours.

        Returns
        -------
        int
            The length of the time steps in hours.

        """
        da_dt = self._ds["time"].diff("time")
        return (da_dt.dt.seconds[0] // 3600).item()

    def get_vars_units(self, category: str) -> List[str]:
        """Return the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[str]
            The units of the variables in the given category.

        """
        if category not in self._ds and category == "forcing":
            warnings.warn("no forcing data found in datastore")
            return []
        return self._ds[f"{category}_feature_units"].values.tolist()

    def get_vars_names(self, category: str) -> List[str]:
        """Return the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[str]
            The names of the variables in the given category.

        """
        if category not in self._ds and category == "forcing":
            warnings.warn("no forcing data found in datastore")
            return []
        return self._ds[f"{category}_feature"].values.tolist()

    def get_vars_long_names(self, category: str) -> List[str]:
        """
        Return the long names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[str]
            The long names of the variables in the given category.

        """
        if category not in self._ds and category == "forcing":
            warnings.warn("no forcing data found in datastore")
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
        self, category: str, split: str, standardize: bool = False
    ) -> xr.DataArray:
        """
        Return the processed data (as a single `xr.DataArray`) for the given
        category of data and test/train/val-split that covers all the data (in
        space and time) of a given category (state/forcin g/static). "state" is
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

        If the data is ensemble data, the dataarray will have an additional
        `ensemble_member` dimension.

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
                **{
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

        """
        ds_unstacked = self.unstack_grid_coords(da_or_ds=self._ds)
        da_state_variable = (
            ds_unstacked["state"].isel(time=0).isel(state_feature=0)
        )
        da_domain_allzero = xr.zeros_like(da_state_variable)
        ds_unstacked["boundary_mask"] = da_domain_allzero.isel(
            x=slice(self._n_boundary_points, -self._n_boundary_points),
            y=slice(self._n_boundary_points, -self._n_boundary_points),
        )
        ds_unstacked["boundary_mask"] = ds_unstacked.boundary_mask.fillna(
            1
        ).astype(int)
        return self.stack_grid_coords(da_or_ds=ds_unstacked.boundary_mask)

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

    @cached_property
    def grid_shape_state(self):
        """The shape of the cartesian grid for the state variables.

        Returns
        -------
        CartesianGridShape
            The shape of the cartesian grid for the state variables.

        """
        ds_state = self.unstack_grid_coords(self._ds["state"])
        da_x, da_y = ds_state.x, ds_state.y
        assert da_x.ndim == da_y.ndim == 1
        return CartesianGridShape(x=da_x.size, y=da_y.size)

    def get_xy(self, category: str, stacked: bool) -> ndarray:
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
            The x, y coordinates of the dataset, returned differently based on
            the value of `stacked`:
            - `stacked==True`: shape `(n_grid_points, 2)` where
                               n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(N_x, N_y, 2)`

        """
        # assume variables are stored in dimensions [grid_index, ...]
        ds_category = self.unstack_grid_coords(da_or_ds=self._ds[category])

        da_xs = ds_category.x
        da_ys = ds_category.y

        assert da_xs.ndim == da_ys.ndim == 1, "x and y coordinates must be 1D"

        da_x, da_y = xr.broadcast(da_xs, da_ys)
        da_xy = xr.concat([da_x, da_y], dim="grid_coord")

        if stacked:
            da_xy = da_xy.stack(grid_index=self.CARTESIAN_COORDS).transpose(
                "grid_index",
                "grid_coord",
            )
        else:
            dims = [
                "x",
                "y",
                "grid_coord",
            ]
            da_xy = da_xy.transpose(*dims)

        return da_xy.values
