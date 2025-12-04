# Interpolate NaN values in IFS zarr dataset at specific timestamps
# For LAM model project
# by Joel Oskarsson, joel.oskarsson@outlook.com

# Standard library
import argparse
import os

# Third-party
import numpy as np
import pandas as pd
import xarray as xr


def interpolate_zarr_at_timestamps(zarr_path):
    # List of (timestamp, prediction_index, level) tuples
    timestamp_pred_level_tuples = [
        ("2020-02-09T12:00:00.000000000", 0, 850),
        ("2020-02-09T12:00:00.000000000", 10, 850),
        ("2020-01-06T12:00:00.000000000", 11, 850),
        ("2019-12-09T12:00:00.000000000", 12, 850),
        ("2020-02-18T12:00:00.000000000", 14, 850),
        ("2020-02-08T12:00:00.000000000", 18, 850),
        ("2020-02-13T00:00:00.000000000", 18, 850),
        ("2020-02-25T12:00:00.000000000", 19, 850),
        ("2019-12-28T12:00:00.000000000", 21, 850),
        ("2020-03-10T12:00:00.000000000", 21, 850),
        ("2020-10-27T00:00:00.000000000", 21, 850),
        ("2020-01-03T00:00:00.000000000", 22, 850),
        ("2020-10-27T12:00:00.000000000", 23, 850),
        ("2020-02-04T00:00:00.000000000", 24, 850),
        ("2020-02-05T12:00:00.000000000", 24, 850),
        ("2020-02-15T00:00:00.000000000", 24, 850),
        ("2020-01-03T00:00:00.000000000", 26, 850),
        ("2020-01-20T00:00:00.000000000", 27, 850),
        ("2020-01-20T12:00:00.000000000", 27, 850),
        ("2020-02-16T12:00:00.000000000", 28, 850),
        ("2020-10-21T00:00:00.000000000", 28, 850),
        ("2020-01-21T00:00:00.000000000", 29, 850),
        ("2020-01-31T12:00:00.000000000", 30, 850),
        ("2020-01-02T00:00:00.000000000", 32, 850),
        ("2020-02-08T00:00:00.000000000", 32, 850),
        ("2020-02-02T00:00:00.000000000", 35, 850),
        ("2020-02-04T00:00:00.000000000", 37, 850),
        ("2019-12-24T12:00:00.000000000", 39, 850),
        ("2020-02-01T00:00:00.000000000", 39, 850),
        ("2020-01-18T00:00:00.000000000", 40, 850),
        ("2020-02-01T12:00:00.000000000", 40, 850),
        ("2020-01-01T12:00:00.000000000", 8, 850),
        ("2020-01-02T00:00:00.000000000", 9, 850),
        ("2020-01-17T12:00:00.000000000", 10, 925),
    ]

    print(f"Opening zarr dataset from: {zarr_path}")
    ds = xr.open_zarr(zarr_path)
    variable_name = "geopotential"
    print(f"Working with variable: {variable_name}")

    # Process each tuple and interpolate NaNs
    for i, (timestamp_str, pred_num, level) in enumerate(
        timestamp_pred_level_tuples
    ):
        timestamp = pd.to_datetime(timestamp_str)
        print(
            f"Processing {i + 1}/{len(timestamp_pred_level_tuples)}: "
            f"Time {timestamp}, Pred {pred_num}, Level {level}"
        )

        # Get the data for this timestamp combination
        data_slice = (
            ds[variable_name]
            .sel(time=[timestamp])
            .isel(prediction_timedelta=[pred_num])
        )

        # Count NaNs before interpolation
        nans_before = np.isnan(data_slice.values).sum()

        if nans_before > 0:
            print(f"  Found {nans_before} NaNs, performing interpolation...")

            # Create a temporary dataset with just this slice
            temp_ds = (
                ds[[variable_name]]
                .sel(time=[timestamp])
                .isel(prediction_timedelta=[pred_num])
            )

            # Interpolate NaNs (use longitude)
            interp_temp_ds = temp_ds.interpolate_na(
                dim="longitude", method="linear"
            )

            # Count NaNs after interpolation
            nans_after = np.isnan(interp_temp_ds[variable_name].values).sum()
            print(f"  After interpolation: {nans_after} NaNs remain")

            # Write chunk back to zarr
            print(interp_temp_ds)
            interp_temp_ds.to_zarr(
                zarr_path, mode="r+", region="auto"
            )  # Rewrite only region
        else:
            print("No NaNs found, skipping interpolation")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate NaN values in dataset at specific timestamps"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the zarr dataset")
    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr directory not found at {args.zarr_path}")
        return

    interpolate_zarr_at_timestamps(args.zarr_path)


if __name__ == "__main__":
    main()
