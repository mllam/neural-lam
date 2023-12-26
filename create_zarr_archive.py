# Standard library
import argparse
import glob
import os
import re
import shutil

import numcodecs

# Third-party
import xarray as xr
from tqdm import tqdm

from neural_lam import constants


def append_or_create_zarr(data_out: xr.Dataset, config: dict, zarr_name: str) -> None:
    """Append data to an existing Zarr archive or create a new one."""

    if config["test_year"] in data_out.time.dt.year.values:
        zarr_path = os.path.join(config["zarr_path"], "test", zarr_name)
    else:
        zarr_path = os.path.join(config["zarr_path"], "train", zarr_name)

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
    """Load weather data from NetCDF files and store it in a Zarr archive."""

    file_paths = []
    for root, dirs, files in os.walk(data_config["data_path"]):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            file_paths.sort()

    # Group file paths into chunks
    file_groups = [
        file_paths[i: i + config["chunk_size"]]
        for i in range(0, len(file_paths),
                       config["chunk_size"])]

    for group in tqdm(file_groups, desc="Processing file groups"):
        # Create a new Zarr archive for each group
        # Extract the date from the first file in the group
        date = os.path.basename(group[0]).split('_')[0][3:]
        zarr_name = f"data_{date}.zarr"
        if not os.path.exists(
            os.path.join(config["zarr_path"],
                         "train", zarr_name)) and not os.path.exists(
            os.path.join(config["zarr_path"],
                         "test", zarr_name)):
            for full_path in group:
                process_file(full_path, config, zarr_name)


def process_file(full_path, config, zarr_name):
    try:
        # if zarr_name directory exists, skip
        match = config["filename_pattern"].match(full_path)
        if not match:
            return None
        data: xr.Dataset = xr.open_dataset(full_path, engine="netcdf4", chunks={
            "time": 1,
            "x_1": -1,
            "y_1": -1,
            "z_1": -1,
            "zbound": -1,
        }, autoclose=True).drop_vars("grid_mapping_1")
        for var in data.variables:
            data[var].encoding = {"compressor": config["compressor"]}
        data.time.encoding = {'dtype': 'float64'}
        append_or_create_zarr(data, config, zarr_name)
        # Display the progress
        print(f"Processed: {full_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}")


def combine_zarr_archives(config) -> None:
    """Combine the last Zarr archive from the train folder with the first from the test
    folder."""

    # Get the last Zarr archive from the train folder
    train_archives = sorted(
        glob.glob(
            os.path.join(
                data_config["zarr_path"],
                "train",
                '*.zarr')))

    # Get the first Zarr archive from the test folder
    test_archives = sorted(
        glob.glob(
            os.path.join(
                data_config["zarr_path"],
                "test",
                '*.zarr')))
    first_test_archive = xr.open_zarr(test_archives[0], consolidated=True)

    val_archives_path = os.path.join(data_config["zarr_path"], "val")

    for t in range(first_test_archive.time.size):
        first_test_archive.isel(time=slice(t, t + 1)).to_zarr(train_archives[-1],
                                                              mode="a",
                                                              append_dim="time",
                                                              consolidated=True)

    shutil.rmtree(test_archives[0])
    shutil.rmtree(test_archives[-1])

    for file in test_archives[1:]:
        filename = os.path.basename(file)
        os.symlink(file, os.path.join(val_archives_path, filename))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create a zarr archive.')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the raw data',
        default="/scratch/mch/sadamov/ml_v1/")
    parser.add_argument('--test_year', type=int, required=True,
                        help='Test year', default=2020)
    parser.add_argument('--filename_regex', type=str, required=True,
                        help='Filename regex', default="(.*)_extr.nc")

    args = parser.parse_args()

    data_config = {
        "data_path": args.data_path,
        "filename_regex": args.filename_regex,
        "zarr_path": "/users/sadamov/pyprojects/neural-cosmo/data/cosmo/samples",
        "compressor": numcodecs.Blosc(
            cname='lz4',
            clevel=7,
            shuffle=numcodecs.Blosc.SHUFFLE),
        "chunk_size": constants.chunk_size,
        "test_year": args.test_year,
    }
    data_config.update(
        {"folders": os.listdir(data_config["data_path"]),
         "filename_pattern": re.compile(data_config["filename_regex"])})

    load_data(data_config)
    combine_zarr_archives(data_config)
