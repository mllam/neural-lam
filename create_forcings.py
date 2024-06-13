# Standard library
import argparse

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# First-party
from neural_lam import config


def get_seconds_in_year(year):
    start_of_year = pd.Timestamp(f"{year}-01-01")
    start_of_next_year = pd.Timestamp(f"{year + 1}-01-01")
    return (start_of_next_year - start_of_year).total_seconds()


def calculate_datetime_forcing(timesteps):
    hours_of_day = xr.DataArray(timesteps.dt.hour, dims=["time"])
    seconds_into_year = xr.DataArray(
        [
            (
                pd.Timestamp(dt_obj)
                - pd.Timestamp(f"{pd.Timestamp(dt_obj).year}-01-01")
            ).total_seconds()
            for dt_obj in timesteps.values
        ],
        dims=["time"],
    )
    year_seconds = xr.DataArray(
        [
            get_seconds_in_year(pd.Timestamp(dt_obj).year)
            for dt_obj in timesteps.values
        ],
        dims=["time"],
    )
    hour_angle = (hours_of_day / 12) * np.pi
    year_angle = (seconds_into_year / year_seconds) * 2 * np.pi
    datetime_forcing = xr.Dataset(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "year_sin": np.sin(year_angle),
            "year_cos": np.cos(year_angle),
        },
        coords={"time": timesteps},
    )
    datetime_forcing = (datetime_forcing + 1) / 2
    return datetime_forcing


def main():
    """Main function for creating the datetime forcing and boundary mask."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config", type=str, default="neural_lam/data_config.yaml"
    )
    parser.add_argument("--zarr_path", type=str, default="data/forcings.zarr")
    args = parser.parse_args()

    data_config = config.Config.from_file(args.data_config)
    dataset = data_config.open_zarrs("state")
    datetime_forcing = calculate_datetime_forcing(timesteps=dataset.time)

    # Expand dimensions to match the target dataset
    datetime_forcing_expanded = datetime_forcing.expand_dims(
        {"y": dataset.y, "x": dataset.x}
    )

    datetime_forcing_expanded = datetime_forcing_expanded.chunk(
        {"time": 1, "y": -1, "x": -1}
    )

    datetime_forcing_expanded.to_zarr(args.zarr_path, mode="w")
    print(f"Datetime forcing saved to {args.zarr_path}")

    dataset


if __name__ == "__main__":
    main()
