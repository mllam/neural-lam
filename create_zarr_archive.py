# Standard library
import os
import re

import numpy as np

# Third-party
import xarray as xr

from neural_lam.constants import data_config


def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None:
    """Append data to an existing Zarr archive or create a new one."""

    if config["test_year"] in data_out.time.dt.year.values:
        zarr_path = os.path.join(config["zarr_path"], "test", "data_test.zarr")
    else:
        zarr_path = os.path.join(config["zarr_path"], "train", "data_train.zarr")

    if os.path.exists(zarr_path):
        if not data_out.time.isin(existing_data).any():
            data_out.to_zarr(
                store=zarr_path,
                mode="a",
                consolidated=True,
                append_dim="time",
            )
    else:
        data_out.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
        )


def load_data(config: dict) -> None:
    """Load weather data from NetCDF files and store it in a Zarr archive.

    The data is assumed to be in a specific directory structure and file naming
    convention, which is checked using regular expressions. The loaded data is chunked
    along the "time" dimension for efficient storage in the Zarr archive.
    If the Zarr archive already exists, new data is appended to it. Otherwise, a new
    Zarr archive is created.
    """
    file_paths = []
    for root, dirs, files in os.walk(data_config["data_path"]):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    for full_path in sorted(file_paths):
        process_file(full_path, config)


def process_file(full_path, config):
    try:
        match = config["filename_pattern"].match(full_path)
        if not match:
            return None
        data: xr.Dataset = xr.open_dataset(full_path, engine="netcdf4", chunks={
            "time": 1,
            "x_1": -1,
            "y_1": -1,
            "z_1": 1,
            "zbound": -1,
        }, autoclose=True)
        for var in data.variables:
            data[var].encoding = {"compressor": config["compressor"]}
        data.time.encoding = {'dtype': 'float64'}
        append_or_create_zarr(data, config)
        # Display the progress
        print(f"Processed: {full_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    data_config.update(
        {"folders": os.listdir(data_config["data_path"]),
         "filename_pattern": re.compile(data_config["filename_regex"])})
    zarr_test = os.path.join(data_config["zarr_path"], "test", "data_test.zarr")
    zarr_train = os.path.join(data_config["zarr_path"], "train", "data_train.zarr")
    # initialize empty np ndarray
    existing_data = []

    if os.path.exists(zarr_train):
        existing_data_train = xr.open_zarr(zarr_train, consolidated=True)
        existing_data.append(existing_data_train.time.values)
    else:
        print("No existing train data found.")
    if os.path.exists(zarr_test):
        existing_data_test = xr.open_zarr(zarr_test, consolidated=True)
        existing_data.append(existing_data_test.time.values)
    else:
        print("No existing test data found.")

    existing_data = np.concatenate(existing_data)

    load_data(data_config)
