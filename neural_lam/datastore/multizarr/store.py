# Standard library
import functools
import os
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Local
from ..base import BaseCartesianDatastore, CartesianGridShape


class MultiZarrDatastore(BaseCartesianDatastore):
    DIMS_TO_KEEP = {"time", "grid_index", "variable_name"}

    def __init__(self, config_path):
        """Create a multi-zarr
        datastore from the
        given configuration
        file. The
        configuration file
        should be a YAML file,
        the format of which is
        should be inferred
        from the example
        configuration file in
        `tests/datastore_examp
        les/multizarr/data_con
        fig.yml`.

        Parameters
        ----------
        config_path : str
            The path to the configuration file.

        """
        self._config_path = Path(config_path)
        self._root_path = self._config_path.parent
        with open(config_path, encoding="utf-8", mode="r") as file:
            self._config = yaml.safe_load(file)

    @property
    def root_path(self):
        """Return the root path of the datastore.

        Returns
        -------
        str
            The root path of the datastore.

        """
        return self._root_path

    def _normalize_path(self, path) -> str:
        """
        Normalize the path of source-dataset defined in the configuration file.
        This assumes that any paths that do not start with a protocol (e.g. `s3://`)
        or are not absolute paths, are relative to the configuration file.

        Parameters
        ----------
        path : str
            The path to normalize.

        Returns
        -------
        str
            The normalized path.
        """
        # try to parse path to see if it defines a protocol, e.g. s3://
        if "://" in path or path.startswith("/"):
            pass
        else:
            # assume path is relative to config file
            path = os.path.join(self._root_path, path)
        return path

    def open_zarrs(self, category):
        """Open the zarr dataset for the given category.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The xarray Dataset object.

        """
        zarr_configs = self._config[category]["zarrs"]

        datasets = []
        for config in zarr_configs:
            dataset_path = self._normalize_path(config["path"])

            try:
                dataset = xr.open_zarr(dataset_path, consolidated=True)
            except Exception as e:
                raise Exception("Error opening dataset:", dataset_path) from e
            datasets.append(dataset)
        merged_dataset = xr.merge(datasets)
        merged_dataset.attrs["category"] = category
        return merged_dataset

    @functools.cached_property
    def coords_projection(self):
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns:
            cartopy.crs.Projection: The projection object.

        """
        proj_config = self._config["projection"]
        proj_class_name = proj_config["class"]
        proj_class = getattr(ccrs, proj_class_name)
        proj_params = proj_config.get("kwargs", {})
        return proj_class(**proj_params)

    @functools.cached_property
    def step_length(self):
        """Return the step length of the dataset in hours.

        Returns:
            int: The step length in hours.

        """
        dataset = self.open_zarrs("state")
        time = dataset.time.isel(time=slice(0, 2)).values
        step_length_ns = time[1] - time[0]
        step_length_hours = step_length_ns / np.timedelta64(1, "h")
        return int(step_length_hours)

    @functools.lru_cache()
    def get_vars_names(self, category):
        """Return the names of the variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            list: The names of the variables in the dataset.

        """
        surface_vars_names = self._config[category].get("surface_vars") or []
        atmosphere_vars_names = [
            f"{var}_{level}"
            for var in (self._config[category].get("atmosphere_vars") or [])
            for level in (self._config[category].get("levels") or [])
        ]
        return surface_vars_names + atmosphere_vars_names

    @functools.lru_cache()
    def get_vars_units(self, category):
        """Return the units of the variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

            Returns:
                list: The units of the variables in the dataset.

        """
        surface_vars_units = self._config[category].get("surface_units") or []
        atmosphere_vars_units = [
            unit
            for unit in (self._config[category].get("atmosphere_units") or [])
            for _ in (self._config[category].get("levels") or [])
        ]
        return surface_vars_units + atmosphere_vars_units

    @functools.lru_cache()
    def get_num_data_vars(self, category):
        """Return the number of data variables in the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            int: The number of data variables in the dataset.

        """
        surface_vars = self._config[category].get("surface_vars", [])
        atmosphere_vars = self._config[category].get("atmosphere_vars", [])
        levels = self._config[category].get("levels", [])

        surface_vars_count = len(surface_vars) if surface_vars is not None else 0
        atmosphere_vars_count = (
            len(atmosphere_vars) if atmosphere_vars is not None else 0
        )
        levels_count = len(levels) if levels is not None else 0

        return surface_vars_count + atmosphere_vars_count * levels_count

    def _stack_grid(self, ds):
        """Stack the grid dimensions of the dataset.

        Args:
            ds (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with stacked grid dimensions.

        """
        if "grid_index" in ds.dims:
            raise ValueError("Grid dimensions already stacked.")
        else:
            if "x" not in ds.dims or "y" not in ds.dims:
                self._rename_dataset_dims_and_vars(dataset=ds)
            ds = ds.stack(grid_index=("y", "x")).reset_index("grid_index")
            # reset the grid_index coordinates to have integer values, otherwise
            # the serialisation to zarr will fail
            ds["grid_index"] = np.arange(len(ds["grid_index"]))
        return ds

    def _convert_dataset_to_dataarray(self, dataset):
        """Convert the Dataset to a Dataarray.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.DataArray: The xarray DataArray object.

        """
        if isinstance(dataset, xr.Dataset):
            dataset = dataset.to_array(dim="variable_name")
        return dataset

    def _filter_dimensions(self, dataset, transpose_array=True):
        """Drop the dimensions and filter the data_vars of the dataset.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            transpose_array (bool): Whether to transpose the array.

        Returns:
            xr.Dataset: The xarray Dataset object with filtered dimensions.
            OR xr.DataArray: The xarray DataArray object with filtered dimensions.

        """
        dims_to_keep = self.DIMS_TO_KEEP
        dataset_dims = set(list(dataset.dims) + ["variable_name"])
        min_req_dims = dims_to_keep.copy()
        min_req_dims.discard("time")
        if not min_req_dims.issubset(dataset_dims):
            missing_dims = min_req_dims - dataset_dims
            print(
                f"\033[91mMissing required dimensions in dataset: "
                f"{missing_dims}\033[0m"
            )
            print(
                "\033[91mAttempting to update dims and "
                "vars based on zarr config...\033[0m"
            )
            dataset = self._rename_dataset_dims_and_vars(
                dataset.attrs["category"], dataset=dataset
            )
            dataset = self._stack_grid(dataset)
            dataset_dims = set(list(dataset.dims) + ["variable_name"])
            if min_req_dims.issubset(dataset_dims):
                print(
                    "\033[92mSuccessfully updated dims and "
                    "vars based on zarr config.\033[0m"
                )
            else:
                print(
                    "\033[91mFailed to update dims and "
                    "vars based on zarr config.\033[0m"
                )
                return None

        dataset_dims = set(list(dataset.dims) + ["variable_name"])
        dims_to_drop = dataset_dims - dims_to_keep
        dataset = dataset.drop_dims(dims_to_drop)
        if dims_to_drop:
            print(
                "\033[91mDropped dimensions: --",
                dims_to_drop,
                "-- from dataset.\033[0m",
            )
            print(
                "\033[91mAny data vars dependent "
                "on these variables were dropped!\033[0m"
            )

        if transpose_array:
            dataset = self._convert_dataset_to_dataarray(dataset)

            if "time" in dataset.dims:
                dataset = dataset.transpose("time", "grid_index", "variable_name")
            else:
                dataset = dataset.transpose("grid_index", "variable_name")
        dataset_vars = (
            list(dataset.data_vars)
            if isinstance(dataset, xr.Dataset)
            else dataset["variable_name"].values.tolist()
        )

        print(  # noqa
            f"\033[94mYour {dataset.attrs['category']} xr.Dataarray has the "
            f"following variables: {dataset_vars} \033[0m",
        )

        return dataset

    def _reshape_grid_to_2d(self, dataset, grid_shape=None):
        """Reshape the grid to 2D for stacked data without multi-index.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            grid_shape (dict): The shape of the grid.

        Returns:
            xr.Dataset: The xarray Dataset object with reshaped grid dimensions.

        """
        if grid_shape is None:
            grid_shape = dict(self.grid_shape_state.values.items())
        x_dim, y_dim = (grid_shape["x"], grid_shape["y"])

        x_coords = np.arange(x_dim)
        y_coords = np.arange(y_dim)
        multi_index = pd.MultiIndex.from_product([y_coords, x_coords], names=["y", "x"])

        mindex_coords = xr.Coordinates.from_pandas_multiindex(multi_index, "grid")
        dataset = dataset.drop_vars(["grid", "x", "y"], errors="ignore")
        dataset = dataset.assign_coords(mindex_coords)
        reshaped_data = dataset.unstack("grid")

        return reshaped_data

    @functools.lru_cache()
    def get_xy(self, category, stacked=True):
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
            The x, y coordinates of the dataset, returned differently based on the
            value of `stacked`:
            - `stacked==True`: shape `(2, n_grid_points)` where n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(2, N_y, N_x)`

        """
        dataset = self.open_zarrs(category)
        xs, ys = dataset.x.values, dataset.y.values

        assert xs.ndim == ys.ndim, "x and y coordinates must have the same dimensions."

        if xs.ndim == 1:
            x, y = np.meshgrid(xs, ys)
        elif x.ndim == 2:
            x, y = xs, ys
        else:
            raise ValueError("Invalid dimensions for x, y coordinates.")

        xy = np.stack((x, y), axis=0)  # (2, N_y, N_x)

        if stacked:
            xy = xy.reshape(2, -1)  # (2, n_grid_points)

        return xy

    @functools.lru_cache()
    def get_normalization_dataarray(self, category: str) -> xr.Dataset:
        """Return the
        normalization
        dataarray for the
        given category. This
        should contain a
        `{category}_mean` and
        `{category}_std`
        variable for each
        variable in the
        category. For
        `category=="state"`,
        the dataarray should
        also contain a
        `state_diff_mean` and
        `state_diff_std`
        variable for the one-
        step differences of
        the state variables.
        The return dataarray
        should at least have
        dimensions of `({categ
        ory}_feature)`, but
        can also include for
        example `grid_index`
        (if the normalisation
        is done per grid point
        for example).

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The normalization dataarray for the given category, with variables
            for the mean and standard deviation of the variables (and
            differences for state variables).

        """
        # XXX: the multizarr code didn't include routines for computing the
        # normalization of "static" features previously, we'll just hack
        # something in here and assume they are already normalized
        if category == "static":
            da_mean = xr.DataArray(
                np.zeros(self.get_num_data_vars(category)),
                dims=("static_feature",),
                coords={"static_feature": self.get_vars_names(category)},
            )
            da_std = xr.DataArray(
                np.ones(self.get_num_data_vars(category)),
                dims=("static_feature",),
                coords={"static_feature": self.get_vars_names(category)},
            )
            return xr.Dataset(dict(static_mean=da_mean, static_std=da_std))

        ds_combined_stats = self._load_and_merge_stats()
        if ds_combined_stats is None:
            return None

        ds_combined_stats = self._rename_data_vars(ds_combined_stats)

        ops = ["mean", "std"]
        stats_variables = [f"{category}_{op}" for op in ops]
        if category == "state":
            stats_variables += [f"state_diff_{op}" for op in ops]

        ds_stats = ds_combined_stats[stats_variables]

        return ds_stats

    def _load_and_merge_stats(self):
        """Load and merge the normalization statistics for the dataset.

        Returns:
            xr.Dataset: The merged normalization statistics for the dataset.

        """
        combined_stats = None
        for i, zarr_config in enumerate(
            self._config["utilities"]["normalization"]["zarrs"]
        ):
            stats_path = self._normalize_path(zarr_config["path"])
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"Normalization statistics not found at path: {stats_path}"
                )
            stats = xr.open_zarr(stats_path, consolidated=True)
            if i == 0:
                combined_stats = stats
            else:
                combined_stats = xr.merge([stats, combined_stats])
        return combined_stats

    def _rename_data_vars(self, combined_stats):
        """Rename the data variables of the normalization statistics.

        Args:
            combined_stats (xr.Dataset): The combined normalization statistics.

        Returns:
            xr.Dataset: The combined normalization statistics with renamed data
            variables.

        """
        vars_mapping = {}
        for zarr_config in self._config["utilities"]["normalization"]["zarrs"]:
            vars_mapping.update(zarr_config["stats_vars"])

        return combined_stats.rename_vars(
            {
                v: k
                for k, v in vars_mapping.items()
                if v in list(combined_stats.data_vars)
            }
        )

    def _select_stats_by_category(self, combined_stats, category):
        """Select the normalization statistics for the given category.

        Args:
            combined_stats (xr.Dataset): The combined normalization statistics.
            category (str): The category of the dataset (state/forcing/static).

        Returns:
            xr.Dataset: The normalization statistics for the dataset.

        """
        if category == "state":
            stats = combined_stats.loc[
                dict(variable_name=self.get_vars_names(category=category))
            ]
            stats = stats.drop_vars(["forcing_mean", "forcing_std"])
            return stats
        elif category == "forcing":
            non_normalized_vars = self.utilities.normalization.non_normalized_vars
            if non_normalized_vars is None:
                non_normalized_vars = []
            forcing_vars = self.vars_names(category)
            normalized_vars = [
                var for var in forcing_vars if var not in non_normalized_vars
            ]
            non_normalized_vars = [
                var for var in forcing_vars if var in non_normalized_vars
            ]
            stats_normalized = combined_stats.loc[
                dict(forcing_variable=normalized_vars)
            ]
            if non_normalized_vars:
                stats_non_normalized = combined_stats.loc[
                    dict(forcing_variable=non_normalized_vars)
                ]
                stats = xr.merge([stats_normalized, stats_non_normalized])
            else:
                stats = stats_normalized
            stats_normalized = stats_normalized[["forcing_mean", "forcing_std"]]

            return stats
        else:
            print(f"Invalid category: {category}")
            return None

    def _extract_vars(self, category, ds=None):
        """Extract (select) the data variables from the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            dataset (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with extracted variables.

        """
        if ds is None:
            ds = self.open_zarrs(category)
        surface_vars = self._config[category].get("surface_vars")
        atmoshere_vars = self._config[category].get("atmosphere_vars")

        ds_surface = None
        if surface_vars is not None:
            ds_surface = ds[surface_vars]

        ds_atmosphere = None
        if atmoshere_vars is not None:
            ds_atmosphere = self._extract_atmosphere_vars(category=category, ds=ds)

        if ds_surface and ds_atmosphere:
            return xr.merge([ds_surface, ds_atmosphere])
        elif ds_surface:
            return ds_surface
        elif ds_atmosphere:
            return ds_atmosphere
        else:
            raise ValueError(f"No variables found in dataset {category}")

    def _extract_atmosphere_vars(self, category, ds):
        """Extract the atmosphere variables from the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            ds (xr.Dataset): The xarray Dataset object.

        Returns:
            xr.Dataset: The xarray Dataset object with atmosphere variables.

        """

        if "level" not in list(ds.dims) and self._config[category]["atmosphere_vars"]:
            ds = self._rename_dataset_dims_and_vars(ds.attrs["category"], dataset=ds)

        data_arrays = [
            ds[var].sel(level=level, drop=True).rename(f"{var}_{level}")
            for var in self._config[category]["atmosphere_vars"]
            for level in self._config[category]["levels"]
        ]

        if self._config[category]["atmosphere_vars"]:
            return xr.merge(data_arrays)
        else:
            return xr.Dataset()

    def _rename_dataset_dims_and_vars(self, category, dataset=None):
        """Rename the dimensions and variables of the dataset.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            dataset (xr.Dataset): The xarray Dataset object. OR xr.DataArray:
            The xarray DataArray object.

        Returns:
            xr.Dataset: The xarray Dataset object with renamed dimensions and
            variables.
            OR xr.DataArray: The xarray DataArray object with renamed
            dimensions and variables.

        """
        convert = False
        if dataset is None:
            dataset = self.open_zarrs(category)
        elif isinstance(dataset, xr.DataArray):
            convert = True
            dataset = dataset.to_dataset("variable_name")
        dims_mapping = {}
        zarr_configs = self._config[category]["zarrs"]
        for zarr_config in zarr_configs:
            dims_mapping.update(zarr_config["dims"])

        dataset = dataset.rename_dims(
            {
                v: k
                for k, v in dims_mapping.items()
                if k not in dataset.dims and v in dataset.dims
            }
        )
        dataset = dataset.rename_vars(
            {v: k for k, v in dims_mapping.items() if v in dataset.coords}
        )
        if convert:
            dataset = dataset.to_array()
        return dataset

    def _apply_time_split(self, dataset, split="train"):
        """Filter the dataset by the time split.

        Args:
            dataset (xr.Dataset): The xarray Dataset object.
            split (str): The time split to filter the dataset.

        Returns:["window"]
            xr.Dataset: The xarray Dataset object filtered by the time split.

        """
        start, end = (
            self._config["splits"][split]["start"],
            self._config["splits"][split]["end"],
        )
        dataset = dataset.sel(time=slice(start, end))
        dataset.attrs["split"] = split
        return dataset

    @property
    def grid_shape_state(self):
        """Return the shape of the state grid.

        Returns:
            CartesianGridShape: The shape of the state grid.

        """
        return CartesianGridShape(
            x=self._config["grid_shape_state"]["x"],
            y=self._config["grid_shape_state"]["y"],
        )

    @property
    def boundary_mask(self) -> xr.DataArray:
        """Load the boundary mask for the dataset, with spatial dimensions stacked.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions `('grid_index',)`.

        """
        boundary_mask_path = self._normalize_path(
            self._config["boundary"]["mask"]["path"]
        )
        ds_boundary_mask = xr.open_zarr(boundary_mask_path)
        return (
            ds_boundary_mask.mask.stack(grid_index=("y", "x"))
            .reset_index("grid_index")
            .astype(int)
        )

    def get_dataarray(self, category, split="train"):
        """Process the dataset for the given category.

        Args:
            category (str): The category of the dataset (state/forcing/static).
            split (str): The time split to filter the dataset (train/val/test).

        Returns:
            xr.DataArray: The xarray DataArray object with processed dataset.

        """
        dataset = self.open_zarrs(category)
        dataset = self._extract_vars(category, dataset)
        if category != "static":
            dataset = self._apply_time_split(dataset, split)
        dataset = self._stack_grid(dataset)
        dataset = self._rename_dataset_dims_and_vars(category, dataset)
        dataset = self._filter_dimensions(dataset)
        dataset = self._convert_dataset_to_dataarray(dataset)
        if category == "static" and "time" in dataset.dims:
            dataset = dataset.isel(time=0, drop=True)

        dataset = dataset.rename(dict(variable_name=f"{category}_feature"))

        return dataset
