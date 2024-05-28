# Standard library
import os

# Third-party
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import yaml


class Config:
    """
    Class for loading configuration files.

    This class loads a YAML configuration file and provides a way to access
    its values as attributes.
    """

    def __init__(self, config_path, values=None):
        self.config_path = config_path
        if values is None:
            self.values = self.load_config()
        else:
            self.values = values

    def load_config(self):
        """Load configuration file."""
        with open(self.config_path, encoding="utf-8", mode="r") as file:
            return yaml.safe_load(file)

    def __getattr__(self, name):
        keys = name.split(".")
        value = self.values
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        if isinstance(value, dict):
            return Config(None, values=value)
        return value

    def __getitem__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return Config(None, values=value)
        return value

    def __contains__(self, key):
        return key in self.values

    def param_names(self):
        """Return parameter names."""
        surface_names = self.values["state"]["surface"]
        atmosphere_names = [
            f"{var}_{level}"
            for var in self.values["state"]["atmosphere"]
            for level in self.values["state"]["levels"]
        ]
        return surface_names + atmosphere_names

    def param_units(self):
        """Return parameter units."""
        surface_units = self.values["state"]["surface_units"]
        atmosphere_units = [
            unit
            for unit in self.values["state"]["atmosphere_units"]
            for _ in self.values["state"]["levels"]
        ]
        return surface_units + atmosphere_units

    def num_data_vars(self, key):
        """Return the number of data variables for a given key."""
        surface_vars = len(self.values[key]["surface"])
        atmosphere_vars = len(self.values[key]["atmosphere"])
        levels = len(self.values[key]["levels"])
        return surface_vars + atmosphere_vars * levels

    def projection(self):
        """Return the projection."""
        proj_config = self.values["projections"]["class"]
        proj_class = getattr(ccrs, proj_config["proj_class"])
        proj_params = proj_config["proj_params"]
        return proj_class(**proj_params)

    def open_zarr(self, dataset_name):
        """Open a dataset specified by the dataset name."""
        dataset_path = self.zarrs[dataset_name].path
        if dataset_path is None or not os.path.exists(dataset_path):
            print(
                f"Dataset '{dataset_name}' "
                f"not found at path: {dataset_path}"
            )
            return None
        dataset = xr.open_zarr(dataset_path, consolidated=True)
        return dataset

    def load_normalization_stats(self):
        """Load normalization statistics from Zarr archive."""
        normalization_path = self.normalization.zarr
        if not os.path.exists(normalization_path):
            print(
                f"Normalization statistics not found at "
                f"path: {normalization_path}"
            )
            return None
        normalization_stats = xr.open_zarr(
            normalization_path, consolidated=True
        )
        return normalization_stats

    def process_dataset(self, dataset_name, split="train", stack=True):
        """Process a single dataset specified by the dataset name."""

        dataset = self.open_zarr(dataset_name)
        if dataset is None:
            return None

        start, end = (
            self.splits[split].start,
            self.splits[split].end,
        )
        dataset = dataset.sel(time=slice(start, end))
        dataset = dataset.rename_dims(
            {
                v: k
                for k, v in self.zarrs[dataset_name].dims.values.items()
                if k not in dataset.dims
            }
        )

        vars_surface = []
        if self[dataset_name].surface:
            vars_surface = dataset[self[dataset_name].surface]

        vars_atmosphere = []
        if self[dataset_name].atmosphere:
            vars_atmosphere = xr.merge(
                [
                    dataset[var]
                    .sel(level=level, drop=True)
                    .rename(f"{var}_{level}")
                    for var in self[dataset_name].atmosphere
                    for level in self[dataset_name].levels
                ]
            )

        if vars_surface and vars_atmosphere:
            dataset = xr.merge([vars_surface, vars_atmosphere])
        elif vars_surface:
            dataset = vars_surface
        elif vars_atmosphere:
            dataset = vars_atmosphere
        else:
            print(f"No variables found in dataset {dataset_name}")
            return None

        if not all(
            lat_lon in self.zarrs[dataset_name].dims.values.values()
            for lat_lon in self.zarrs[
                dataset_name
            ].lat_lon_names.values.values()
        ):
            lat_name = self.zarrs[dataset_name].lat_lon_names.lat
            lon_name = self.zarrs[dataset_name].lat_lon_names.lon
            if dataset[lat_name].ndim == 2:
                dataset[lat_name] = dataset[lat_name].isel(x=0, drop=True)
            if dataset[lon_name].ndim == 2:
                dataset[lon_name] = dataset[lon_name].isel(y=0, drop=True)
            dataset = dataset.assign_coords(
                x=dataset[lon_name], y=dataset[lat_name]
            )

        if stack:
            dataset = self.stack_grid(dataset)

        return dataset

    def stack_grid(self, dataset):
        """Stack grid dimensions."""
        dataset = dataset.squeeze().stack(grid=("x", "y")).to_array()

        if "time" in dataset.dims:
            dataset = dataset.transpose("time", "grid", "variable")
        else:
            dataset = dataset.transpose("grid", "variable")
        return dataset

    def get_nwp_xy(self):
        """Get the x and y coordinates for the NWP grid."""
        x = self.process_dataset("static", stack=False).x.values
        y = self.process_dataset("static", stack=False).y.values
        xx, yy = np.meshgrid(y, x)
        xy = np.stack((xx, yy), axis=0)

        return xy
