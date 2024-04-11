# Standard library
import argparse
import os
import re

# Third-party
import numcodecs
import xarray as xr
from tqdm import tqdm


def create_single_zarr_archive(config: dict, is_test: bool) -> None:
    """
    Create a single large Zarr archive for either test or train data.
    """
    # Determine the path based on whether it's test or train data
    zarr_path = os.path.join(
        config["zarr_path"], "test" if is_test else "train"
    )
    zarr_name = "test_data.zarr" if is_test else "train_data.zarr"
    full_zarr_path = os.path.join(zarr_path, zarr_name)

    # Ensure the directory exists
    os.makedirs(zarr_path, exist_ok=True)

    # Initialize an empty list to store datasets
    datasets = []

    # Loop through all files and process
    for root, _, files in os.walk(config["data_path"]):
        for file in tqdm(files, desc="Processing files"):
            full_path = os.path.join(root, file)
            match = config["filename_pattern"].match(file)
            if not match:
                continue

            # Open the dataset
            data = xr.open_dataset(
                full_path,
                engine="netcdf4",
                chunks={"time": 1},  # Chunk only along the time dimension
                autoclose=True,
            ).drop_vars("grid_mapping_1", errors="ignore")

            # Check if the data belongs to the test year
            data_is_test = config["test_year"] in data.time.dt.year.values

            # If the current data matches the desired type (test/train)
            if data_is_test == is_test:
                datasets.append(data)

    # Combine all datasets along the time dimension
    combined_data = xr.concat(datasets, dim="time")

    # Set optimal compression
    for var in combined_data.variables:
        combined_data[var].encoding = {"compressor": config["compressor"]}

    # Save the combined dataset to a Zarr archive
    combined_data.to_zarr(
        store=full_zarr_path,
        mode="w",
        consolidated=True,
    )
    print(f"Created Zarr archive at {full_zarr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Zarr archives for weather data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/scratch/mch/sadamov/ml_v1/",
        help="Path to the raw data",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="data/cosmo/samples/",
        help="Path to the zarr output",
    )
    parser.add_argument("--test_year", type=int, default=2020)
    parser.add_argument(
        "--filename_regex",
        type=str,
        help="Filename regex",
        default="(.*)_extr.nc",
    )

    args = parser.parse_args()

    data_config = {
        "data_path": args.data_path,
        "filename_regex": args.filename_regex,
        "zarr_path": args.zarr_path,
        "compressor": numcodecs.Blosc(
            cname="lz4", clevel=7, shuffle=numcodecs.Blosc.SHUFFLE
        ),
        "test_year": args.test_year,
        "filename_pattern": re.compile(args.filename_regex),
    }

    # Create Zarr archive for test data
    create_single_zarr_archive(data_config, is_test=True)

    # Create Zarr archive for train data
    create_single_zarr_archive(data_config, is_test=False)
