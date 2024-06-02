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


class Config:
    DIMS_TO_KEEP = {"time", "grid", "variable"}

    def __init__(self, values):
        self.values = values

    @classmethod
    def from_file(cls, filepath):
        if filepath.endswith(".yaml"):
            with open(filepath, encoding="utf-8", mode="r") as file:
                return cls(values=yaml.safe_load(file))
        else:
            raise NotImplementedError(Path(filepath).suffix)

    def __getattr__(self, name):
        keys = name.split(".")
        value = self.values
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        if isinstance(value, dict):
            return Config(values=value)
        return value

    def __getitem__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return Config(values=value)
        return value

    def __contains__(self, key):
        return key in self.values

    @functools.cached_property
    def coords_projection(self):
        """Return the projection object for the coordinates."""
        proj_config = self.values["projection"]
        proj_class_name = proj_config["class"]
        proj_class = getattr(ccrs, proj_class_name)
        proj_params = proj_config.get("kwargs", {})
        return proj_class(**proj_params)

    @functools.lru_cache()
    def vars_names(self, category):
        """Return the names of the variables in the dataset."""
        surface_vars_names = self.values[category].get("surface_vars") or []
        atmosphere_vars_names = [
            f"{var}_{level}"
            for var in (self.values[category].get("atmosphere_vars") or [])
            for level in (self.values[category].get("levels") or [])
        ]
        return surface_vars_names + atmosphere_vars_names

    @functools.lru_cache()
    def vars_units(self, category):
        """Return the units of the variables in the dataset."""
        surface_vars_units = self.values[category].get("surface_units") or []
        atmosphere_vars_units = [
            unit
            for unit in (self.values[category].get("atmosphere_units") or [])
            for _ in (self.values[category].get("levels") or [])
        ]
        return surface_vars_units + atmosphere_vars_units

    @functools.lru_cache()
    def num_data_vars(self, category):
        """Return the number of data variables in the dataset."""
        surface_vars = self.values[category].get("surface_vars", [])
        atmosphere_vars = self.values[category].get("atmosphere_vars", [])
        levels = self.values[category].get("levels", [])

        surface_vars_count = (
            len(surface_vars) if surface_vars is not None else 0
        )
        atmosphere_vars_count = (
            len(atmosphere_vars) if atmosphere_vars is not None else 0
        )
        levels_count = len(levels) if levels is not None else 0

        return surface_vars_count + atmosphere_vars_count * levels_count

    def open_zarr(self, category):
        """Open the zarr dataset for the given category."""
        zarr_configs = self.values[category]["zarrs"]

        try:
            datasets = []
            for config in zarr_configs:
                dataset_path = config["path"]
                dataset = xr.open_zarr(dataset_path, consolidated=True)
                datasets.append(dataset)
            merged_dataset = xr.merge(datasets)
            merged_dataset.attrs["category"] = category
            return merged_dataset
        except Exception:
            print(f"Invalid zarr configuration for category: {category}")
            return None

    def stack_grid(self, dataset):
        """Stack the grid dimensions of the dataset."""
        if dataset is None:
            return None
        dims = dataset.to_array().dims

        if "grid" in dims:
            print("\033[94mGrid dimensions already stacked.\033[0m")
            return dataset.squeeze()
        else:
            if "x" not in dims or "y" not in dims:
                self.rename_dataset_dims_and_vars(dataset=dataset)
            dataset = dataset.squeeze().stack(grid=("x", "y"))
        return dataset

    def convert_dataset_to_dataarray(self, dataset):
        """Convert the Dataset to a Dataarray."""
        if isinstance(dataset, xr.Dataset):
            dataset = dataset.to_array()
        return dataset

    def filter_dimensions(self, dataset, transpose_array=True):
        """Filter the dimensions of the dataset."""
        dims_to_keep = self.DIMS_TO_KEEP
        dataset_dims = set(dataset.to_array().dims)
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
            dataset = self.rename_dataset_dims_and_vars(
                dataset.attrs["category"], dataset=dataset
            )
            dataset = self.stack_grid(dataset)
            dataset_dims = set(dataset.to_array().dims)
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

        dataset_dims = set(dataset.to_array().dims)
        dims_to_drop = dataset_dims - dims_to_keep
        dataset = dataset.drop_dims(dims_to_drop)
        if dims_to_drop:
            print(
                "\033[91mDropped dimensions: --",
                dims_to_drop,
                "-- from dataset.\033[0m",
            )
            print(
                "\033[91mAny data vars still dependent "
                "on these variables were dropped!\033[0m"
            )

        if transpose_array:
            dataset = self.convert_dataset_to_dataarray(dataset)

            if "time" in dataset.dims:
                dataset = dataset.transpose("time", "grid", "variable")
            else:
                dataset = dataset.transpose("grid", "variable")
        dataset_vars = (
            list(dataset.data_vars)
            if isinstance(dataset, xr.Dataset)
            else dataset["variable"].values.tolist()
        )
        print(
            "\033[94mYour Dataarray has the following variables: ",
            dataset_vars,
            "\033[0m",
        )

        return dataset

    def reshape_grid_to_2d(self, dataset, grid_shape=None):
        """Reshape the grid to 2D."""
        if grid_shape is None:
            grid_shape = dict(self.grid_shape_state.values.items())
        x_dim, y_dim = (grid_shape["x"], grid_shape["y"])

        x_coords = np.arange(x_dim)
        y_coords = np.arange(y_dim)
        multi_index = pd.MultiIndex.from_product(
            [x_coords, y_coords], names=["x", "y"]
        )

        mindex_coords = xr.Coordinates.from_pandas_multiindex(
            multi_index, "grid"
        )
        dataset = dataset.drop_vars(["grid", "x", "y"], errors="ignore")
        dataset = dataset.assign_coords(mindex_coords)
        reshaped_data = dataset.unstack("grid")

        return reshaped_data

    @functools.lru_cache()
    def get_xy(self, category):
        """Return the x, y coordinates of the dataset."""
        dataset = self.open_zarr(category)
        x, y = dataset.x.values, dataset.y.values
        if x.ndim == 1:
            x, y = np.meshgrid(y, x)
        xy = np.stack((x, y), axis=0)

        return xy

    @functools.lru_cache()
    def load_normalization_stats(self, category):
        """Load the normalization statistics for the dataset."""
        for i, zarr_config in enumerate(
            self.values["utilities"]["normalization"]["zarrs"]
        ):
            stats_path = zarr_config["path"]
            if not os.path.exists(stats_path):
                print(
                    f"Normalization statistics not found at path: "
                    f"{stats_path}"
                )
                return None
            stats = xr.open_zarr(stats_path, consolidated=True)
            if i == 0:
                combined_stats = stats
            else:
                stats = xr.merge([stats, combined_stats])
                combined_stats = stats

        # Rename data variables
        vars_mapping = {}
        zarr_configs = self.values["utilities"]["normalization"]["zarrs"]
        for zarr_config in zarr_configs:
            vars_mapping.update(zarr_config["stats_vars"])

        combined_stats = combined_stats.rename_vars(
            {
                v: k
                for k, v in vars_mapping.items()
                if v in list(combined_stats.data_vars)
            }
        )

        stats = combined_stats.loc[dict(variable=self.vars_names(category))]
        if category == "state":
            stats = stats.drop_vars(["forcing_mean", "forcing_std"])
        elif category == "forcing":
            stats = stats[["forcing_mean", "forcing_std"]]
        else:
            print(f"Invalid category: {category}")
            return None
        return stats

    # def assign_lat_lon_coords(self, category, dataset=None):
    #     """Process the latitude and longitude names of the dataset."""
    #     if dataset is None:
    #         dataset = self.open_zarr(category)
    #     lat_lon_names = {}
    #     for zarr_config in self.values[category]["zarrs"]:
    #         lat_lon_names.update(zarr_config["lat_lon_names"])
    #     lat_name, lon_name = (lat_lon_names["lat"], lat_lon_names["lon"])

    #     if "x" not in dataset.dims or "y" in dataset.dims:
    #         dataset = self.reshape_grid_to_2d(dataset)
    #     if not set(lat_lon_names).issubset(dataset.to_array().dims):
    #         dataset = dataset.assign_coords(
    #             x=dataset[lon_name], y=dataset[lat_name]
    #         )
    #     return dataset

    def extract_vars(self, category, dataset=None):
        """Extract the variables from the dataset."""
        if dataset is None:
            dataset = self.open_zarr(category)
        surface_vars = (
            dataset[self[category].surface_vars]
            if self[category].surface_vars
            else []
        )

        if (
            "level" not in dataset.to_array().dims
            and self[category].atmosphere_vars
        ):
            dataset = self.rename_dataset_dims_and_vars(
                dataset.attrs["category"], dataset=dataset
            )

        atmosphere_vars = (
            xr.merge(
                [
                    dataset[var]
                    .sel(level=level, drop=True)
                    .rename(f"{var}_{level}")
                    for var in self[category].atmosphere_vars
                    for level in self[category].levels
                ]
            )
            if self[category].atmosphere_vars
            else []
        )

        if surface_vars and atmosphere_vars:
            return xr.merge([surface_vars, atmosphere_vars])
        elif surface_vars:
            return surface_vars
        elif atmosphere_vars:
            return atmosphere_vars
        else:
            print(f"No variables found in dataset {category}")
            return None

    def rename_dataset_dims_and_vars(self, category, dataset=None):
        """Rename the dimensions and variables of the dataset."""
        convert = False
        if dataset is None:
            dataset = self.open_zarr(category)
        elif isinstance(dataset, xr.DataArray):
            convert = True
            dataset = dataset.to_dataset("variable")
        dims_mapping = {}
        zarr_configs = self.values[category]["zarrs"]
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

    def filter_dataset_by_time(self, dataset, split="train"):
        """Filter the dataset by the time split."""
        start, end = (
            self.values["splits"][split]["start"],
            self.values["splits"][split]["end"],
        )
        return dataset.sel(time=slice(start, end))

    def process_dataset(self, category, split="train"):
        """Process the dataset for the given category."""
        dataset = self.open_zarr(category)
        dataset = self.extract_vars(category, dataset)
        dataset = self.filter_dataset_by_time(dataset, split)
        dataset = self.stack_grid(dataset)
        dataset = self.rename_dataset_dims_and_vars(category, dataset)
        dataset = self.filter_dimensions(dataset)
        dataset = self.convert_dataset_to_dataarray(dataset)

        return dataset
