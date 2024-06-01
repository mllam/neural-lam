# Standard library
import functools
import os
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import yaml


class Config:
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
        proj_config = self.values["projection"]
        proj_class_name = proj_config["class"]
        proj_class = getattr(ccrs, proj_class_name)
        proj_params = proj_config.get("kwargs", {})
        return proj_class(**proj_params)

    @functools.cached_property
    def param_names(self):
        surface_vars_names = self.values["state"]["surface_vars"]
        atmosphere_vars_names = [
            f"{var}_{level}"
            for var in self.values["state"]["atmosphere_vars"]
            for level in self.values["state"]["levels"]
        ]
        return surface_vars_names + atmosphere_vars_names

    @functools.cached_property
    def param_units(self):
        surface_vars_units = self.values["state"]["surface_units"]
        atmosphere_vars_units = [
            unit
            for unit in self.values["state"]["atmosphere_units"]
            for _ in self.values["state"]["levels"]
        ]
        return surface_vars_units + atmosphere_vars_units

    @functools.lru_cache()
    def num_data_vars(self, category):
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

    @functools.lru_cache(maxsize=None)
    def open_zarr(self, category):
        zarr_configs = self.values[category]["zarrs"]

        try:
            datasets = []
            for config in zarr_configs:
                dataset_path = config["path"]
                dataset = xr.open_zarr(dataset_path, consolidated=True)
                datasets.append(dataset)
            return xr.merge(datasets)
        except Exception:
            print(f"Invalid zarr configuration for category: {category}")
            return None

    def stack_grid(self, dataset):
        dims = dataset.to_array().dims

        if "grid" not in dims and "x" in dims and "y" in dims:
            dataset = dataset.squeeze().stack(grid=("x", "y")).to_array()
        else:
            try:
                dataset = dataset.squeeze().to_array()
            except ValueError:
                print("Failed to stack grid dimensions.")
                return None

        if "time" in dataset.dims:
            dataset = dataset.transpose("time", "grid", "variable")
        else:
            dataset = dataset.transpose("grid", "variable")
        return dataset

    @functools.lru_cache()
    def get_nwp_xy(self, category):
        dataset = self.open_zarr(category)
        lon_name = self.values[category]["zarrs"][0]["lat_lon_names"]["lon"]
        lat_name = self.values[category]["zarrs"][0]["lat_lon_names"]["lat"]
        if lon_name in dataset and lat_name in dataset:
            lon = dataset[lon_name].values
            lat = dataset[lat_name].values
        else:
            raise ValueError(
                f"Dataset does not contain " f"{lon_name} or {lat_name}"
            )
        if lon.ndim == 1:
            lon, lat = np.meshgrid(lat, lon)
        lonlat = np.stack((lon.T, lat.T), axis=0)

        return lonlat

    @functools.cached_property
    def load_normalization_stats(self):
        for i, zarr_config in enumerate(self.values["normalization"]["zarrs"]):
            normalization_path = zarr_config["path"]
            if not os.path.exists(normalization_path):
                print(
                    f"Normalization statistics not found at path: "
                    f"{normalization_path}"
                )
                return None
            stats = xr.open_zarr(normalization_path, consolidated=True)
            if i == 0:
                normalization_stats = stats
            else:
                stats = xr.merge([stats, normalization_stats])
                normalization_stats = stats
        return normalization_stats

    @functools.lru_cache(maxsize=None)
    def process_dataset(self, category, split="train"):
        dataset = self.open_zarr(category)
        if dataset is None:
            return None

        start, end = (
            self.values["splits"][split]["start"],
            self.values["splits"][split]["end"],
        )
        dataset = dataset.sel(time=slice(start, end))

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

        surface_vars = []
        if self[category].surface_vars:
            surface_vars = dataset[self[category].surface_vars]

        atmosphere_vars = []
        if self[category].atmosphere_vars:
            atmosphere_vars = xr.merge(
                [
                    dataset[var]
                    .sel(level=level, drop=True)
                    .rename(f"{var}_{level}")
                    for var in self[category].atmosphere_vars
                    for level in self[category].levels
                ]
            )

        if surface_vars and atmosphere_vars:
            dataset = xr.merge([surface_vars, atmosphere_vars])
        elif surface_vars:
            dataset = surface_vars
        elif atmosphere_vars:
            dataset = atmosphere_vars
        else:
            print(f"No variables found in dataset {category}")
            return None

        lat_lon_names = {}
        for zarr_config in self.values[category]["zarrs"]:
            lat_lon_names.update(zarr_config["lat_lon_names"])

        if not all(
            lat_lon in lat_lon_names.values() for lat_lon in lat_lon_names
        ):
            lat_name, lon_name = list(lat_lon_names.values())[:2]
            if dataset[lat_name].ndim == 2:
                dataset[lat_name] = dataset[lat_name].isel(x=0, drop=True)
            if dataset[lon_name].ndim == 2:
                dataset[lon_name] = dataset[lon_name].isel(y=0, drop=True)
            dataset = dataset.assign_coords(
                x=dataset[lon_name], y=dataset[lat_name]
            )

        dataset = dataset.rename(
            {v: k for k, v in dims_mapping.items() if v in dataset.coords}
        )
        dataset = self.stack_grid(dataset)
        return dataset
