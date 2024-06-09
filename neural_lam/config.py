# Standard library
import functools
import os
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import torch
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

    @functools.cached_property
    def step_length(self):
        """Return the step length of the dataset in hours."""
        dataset = self.open_zarrs("state")
        time = dataset.time.isel(time=slice(0, 2)).values
        step_length_ns = time[1] - time[0]
        step_length_hours = step_length_ns / np.timedelta64(1, "h")
        return int(step_length_hours)

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

    def open_zarrs(self, category):
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
        if dataset is None:
            return None
        dims = list(dataset.dims)

        if "grid" in dims:
            print("\033[94mGrid dimensions already stacked.\033[0m")
            return dataset.squeeze()
        else:
            if "x" not in dims or "y" not in dims:
                self.rename_dataset_dims_and_vars(dataset=dataset)
            dataset = dataset.squeeze().stack(grid=("y", "x"))
        return dataset

    def convert_dataset_to_dataarray(self, dataset):
        """Convert the Dataset to a Dataarray."""
        if isinstance(dataset, xr.Dataset):
            dataset = dataset.to_array()
        return dataset

    def filter_dimensions(self, dataset, transpose_array=True):
        """Filter the dimensions of the dataset."""
        dims_to_keep = self.DIMS_TO_KEEP
        dataset_dims = set(list(dataset.dims) + ["variable"])
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
            dataset_dims = set(list(dataset.dims) + ["variable"])
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

        dataset_dims = set(list(dataset.dims) + ["variable"])
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

        print(  # noqa
            f"\033[94mYour {dataset.attrs['category']} xr.Dataarray has the "
            f"following variables: {dataset_vars} \033[0m",
        )

        return dataset

    def reshape_grid_to_2d(self, dataset, grid_shape=None):
        """Reshape the grid to 2D for stacked data without multi-index."""
        if grid_shape is None:
            grid_shape = dict(self.grid_shape_state.values.items())
        x_dim, y_dim = (grid_shape["x"], grid_shape["y"])

        x_coords = np.arange(x_dim)
        y_coords = np.arange(y_dim)
        multi_index = pd.MultiIndex.from_product(
            [y_coords, x_coords], names=["y", "x"]
        )

        mindex_coords = xr.Coordinates.from_pandas_multiindex(
            multi_index, "grid"
        )
        dataset = dataset.drop_vars(["grid", "x", "y"], errors="ignore")
        dataset = dataset.assign_coords(mindex_coords)
        reshaped_data = dataset.unstack("grid")

        return reshaped_data

    @functools.lru_cache()
    def get_xy(self, category, stacked=True):
        """Return the x, y coordinates of the dataset."""
        dataset = self.open_zarrs(category)
        x, y = dataset.x.values, dataset.y.values
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        if stacked:
            xy = np.stack((x, y), axis=0)  # (2, N_y, N_x)
            return xy
        return x, y

    def get_xy_extent(self, category):
        """Return the extent of the x, y coordinates."""
        x, y = self.get_xy(category, stacked=False)
        if self.projection.inverted:
            extent = [x.max(), x.min(), y.max(), y.min()]
        else:
            extent = [x.min(), x.max(), y.min(), y.max()]

        return extent

    @functools.lru_cache()
    def load_normalization_stats(self, category, datatype="torch"):
        """Load the normalization statistics for the dataset."""
        combined_stats = self._load_and_merge_stats()
        if combined_stats is None:
            return None

        combined_stats = self._rename_data_vars(combined_stats)

        stats = self._select_stats_by_category(combined_stats, category)
        if stats is None:
            return None

        if datatype == "torch":
            return self._convert_stats_to_torch(stats)

        return stats

    def _load_and_merge_stats(self):
        combined_stats = None
        for i, zarr_config in enumerate(
            self.values["utilities"]["normalization"]["zarrs"]
        ):
            stats_path = zarr_config["path"]
            if not os.path.exists(stats_path):
                print(
                    f"Normalization statistics not found at path: {stats_path}"
                )
                return None
            stats = xr.open_zarr(stats_path, consolidated=True)
            if i == 0:
                combined_stats = stats
            else:
                combined_stats = xr.merge([stats, combined_stats])
        return combined_stats

    def _rename_data_vars(self, combined_stats):
        vars_mapping = {}
        for zarr_config in self.values["utilities"]["normalization"]["zarrs"]:
            vars_mapping.update(zarr_config["stats_vars"])

        return combined_stats.rename_vars(
            {
                v: k
                for k, v in vars_mapping.items()
                if v in list(combined_stats.data_vars)
            }
        )

    def _select_stats_by_category(self, combined_stats, category):
        if category == "state":
            stats = combined_stats.loc[dict(variable=self.vars_names(category))]
            stats = stats.drop_vars(["forcing_mean", "forcing_std"])
            return stats
        elif category == "forcing":
            non_normalized_vars = (
                self.utilities.normalization.non_normalized_vars
            )
            if non_normalized_vars is None:
                non_normalized_vars = []
            vars = self.vars_names(category)
            window = self["forcing"]["window"]
            forcing_vars = [f"{var}_{i}" for var in vars for i in range(window)]
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

    def _convert_stats_to_torch(self, stats):
        return {
            var: torch.tensor(stats[var].values, dtype=torch.float32)
            for var in stats.data_vars
        }

    def extract_vars(self, category, dataset=None):
        if dataset is None:
            dataset = self.open_zarrs(category)
        surface_vars = None
        atmosphere_vars = None
        if self[category].surface_vars:
            surface_vars = self._extract_surface_vars(category, dataset)
        if self[category].atmosphere_vars:
            atmosphere_vars = self._extract_atmosphere_vars(category, dataset)
        if surface_vars and atmosphere_vars:
            return xr.merge([surface_vars, atmosphere_vars])
        elif surface_vars:
            return surface_vars
        elif atmosphere_vars:
            return atmosphere_vars
        else:
            print(f"No variables found in dataset {category}")
            return None

    def _extract_surface_vars(self, category, dataset):
        return (
            dataset[self[category].surface_vars]
            if self[category].surface_vars
            else []
        )

    def _extract_atmosphere_vars(self, category, dataset):
        if "level" not in list(dataset.dims) and self[category].atmosphere_vars:
            dataset = self.rename_dataset_dims_and_vars(
                dataset.attrs["category"], dataset=dataset
            )

        data_arrays = [
            dataset[var].sel(level=level, drop=True).rename(f"{var}_{level}")
            for var in self[category].atmosphere_vars
            for level in self[category].levels
        ]

        if self[category].atmosphere_vars:
            return xr.merge(data_arrays)
        else:
            return xr.Dataset()

    def rename_dataset_dims_and_vars(self, category, dataset=None):
        """Rename the dimensions and variables of the dataset."""
        convert = False
        if dataset is None:
            dataset = self.open_zarrs(category)
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
        dataset = dataset.sel(time=slice(start, end))
        dataset.attrs["split"] = split
        return dataset

    def apply_window(self, category, dataset=None):
        """Apply the forcing window to the forcing dataset."""
        if dataset is None:
            dataset = self.open_zarrs(category)
        if isinstance(dataset, xr.Dataset):
            dataset = self.convert_dataset_to_dataarray(dataset)
        state = self.open_zarrs("state")
        state = self.filter_dataset_by_time(state, dataset.attrs["split"])
        state_time = state.time.values
        window = self[category].window
        dataset = (
            dataset.sel(time=state_time, method="nearest")
            .pad(time=(window // 2, window // 2), mode="edge")
            .rolling(time=window, center=True)
            .construct("window")
            .stack(variable_window=("variable", "window"))
        )
        dataset = dataset.isel(time=slice(window // 2, -window // 2 + 1))
        return dataset

    def load_boundary_mask(self):
        """Load the boundary mask for the dataset."""
        boundary_mask = xr.open_zarr(self.values["boundary"]["mask"]["path"])
        return torch.tensor(
            boundary_mask.mask.stack(grid=("y", "x")).values,
            dtype=torch.float32,
        ).unsqueeze(1)

    def process_dataset(self, category, split="train", apply_windowing=True):
        """Process the dataset for the given category."""
        dataset = self.open_zarrs(category)
        dataset = self.extract_vars(category, dataset)
        if category != "static":
            dataset = self.filter_dataset_by_time(dataset, split)
        dataset = self.stack_grid(dataset)
        dataset = self.rename_dataset_dims_and_vars(category, dataset)
        dataset = self.filter_dimensions(dataset)
        dataset = self.convert_dataset_to_dataarray(dataset)
        if "window" in self.values[category] and apply_windowing:
            dataset = self.apply_window(category, dataset)
        if category == "static" and "time" in dataset.dims:
            dataset = dataset.isel(time=0, drop=True)

        return dataset
