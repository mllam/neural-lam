# Standard library
import argparse
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# First-party
from neural_lam.datastore.multizarr import MultiZarrDatastore


def get_seconds_in_year(year):
    start_of_year = pd.Timestamp(f"{year}-01-01")
    start_of_next_year = pd.Timestamp(f"{year + 1}-01-01")
    return (start_of_next_year - start_of_year).total_seconds()


def calculate_datetime_forcing(da_time: xr.DataArray):
    """Compute the datetime forcing for a given set of timesteps, assuming that
    timesteps is a DataArray with a type of `np.datetime64`.

    Parameters
    ----------
    timesteps : xr.DataArray
        The timesteps for which to compute the datetime forcing.

    Returns
    -------
    xr.Dataset
        The datetime forcing, with the following variables:
        - hour_sin: The sine of the hour of the day, normalized to [0, 1].
        - hour_cos: The cosine of the hour of the day, normalized to [0, 1].
        - year_sin: The sine of the time of year, normalized to [0, 1].
        - year_cos: The cosine of the time of year, normalized to [0, 1].
    """
    hours_of_day = xr.DataArray(da_time.dt.hour, dims=["time"])
    seconds_into_year = xr.DataArray(
        [
            (
                pd.Timestamp(dt_obj)
                - pd.Timestamp(f"{pd.Timestamp(dt_obj).year}-01-01")
            ).total_seconds()
            for dt_obj in da_time.values
        ],
        dims=["time"],
    )
    year_seconds = xr.DataArray(
        [
            get_seconds_in_year(pd.Timestamp(dt_obj).year)
            for dt_obj in da_time.values
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
        coords={"time": da_time},
    )
    datetime_forcing = (datetime_forcing + 1) / 2
    return datetime_forcing


def main():
    """Main function for creating the datetime forcing and boundary mask."""
    parser = argparse.ArgumentParser(
        description="Create the datetime forcing for neural LAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_config",
        type=str,
        help="Path to data config file",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Path to save the Zarr archive "
        "(default: same directory as the data-config)",
    )
    args = parser.parse_args()

    zarr_path = args.zarr_path
    if zarr_path is None:
        zarr_path = Path(args.data_config).parent / "datetime_forcings.zarr"

    datastore = MultiZarrDatastore(config_path=args.data_config)
    da_state = datastore.get_dataarray(category="state", split="train")

    da_datetime_forcing = calculate_datetime_forcing(
        da_time=da_state.time
    ).expand_dims({"grid_index": da_state.grid_index})

    chunking = {"time": 1}

    if "x" in da_state.coords and "y" in da_state.coords:
        # copy the x and y coordinates to the datetime forcing
        for aux_coord in ["x", "y"]:
            da_datetime_forcing.coords[aux_coord] = da_state[aux_coord]

        da_datetime_forcing = da_datetime_forcing.set_index(
            grid_index=("y", "x")
        ).unstack("grid_index")
        chunking["x"] = -1
        chunking["y"] = -1
    else:
        chunking["grid_index"] = -1

    da_datetime_forcing = da_datetime_forcing.chunk(chunking)

    da_datetime_forcing.to_zarr(zarr_path, mode="w")
    print(da_datetime_forcing)
    print(f"Datetime forcing saved to {zarr_path}")


if __name__ == "__main__":
    main()
