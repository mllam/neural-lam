"""Load weather data from NetCDF files and store it in a Zarr archive."""
# Standard library
import os
import re

import dask.bag

# Third-party
import xarray as xr
from filelock import FileLock

from neural_lam.constants import data_config


def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None:
    """Append data to an existing Zarr archive or create a new one."""

    if config["test_year"] in data_out.time.dt.year.values:
        zarr_path = os.path.join(config["zarr_path"], "test", "data_test.zarr")
    else:
        zarr_path = os.path.join(config["zarr_path"], "train", "data_train.zarr")

    lock = FileLock(zarr_path + ".lock")

    with lock:
        if os.path.exists(zarr_path):
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

    # Create a Dask bag of file paths
    bag = dask.bag.from_sequence(sorted(file_paths))

    def process_file(full_path):
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

    # Map the function across all items in the bag
    bag.map(process_file).compute()


if __name__ == "__main__":
    data_config.update(
        {"folders": os.listdir(data_config["data_path"]),
         "filename_pattern": re.compile(data_config["filename_regex"])})
    load_data(data_config)
