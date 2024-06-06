# Standard library
import os
from argparse import ArgumentParser

# Third-party
import graphcast.data_utils as gc_du
import graphcast.solar_radiation as gc_sr
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xa

# First-party
from neural_lam import vis


def progress_to_sin_cos(progress):
    """
    Transform year/day progress in [0,1] with sin and cos, normalized to [0,1]
    """
    prog_sin = (np.sin(progress * 2 * np.pi) + 1) / 2
    prog_cos = (np.cos(progress * 2 * np.pi) + 1) / 2
    return prog_sin, prog_cos


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="global_example_era5",
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If fields should be plotted " "(default: 0 (false))",
    )
    args = parser.parse_args()

    fields_group_path = os.path.join("data", args.dataset, "fields.zarr")
    fields_group = xa.open_zarr(fields_group_path)
    forcing_path = os.path.join("data", args.dataset, "forcing.zarr")

    # Lat-lon
    grid_lat_vals = np.array(
        fields_group["latitude"], dtype=np.float32
    )  # (num_lat,)
    grid_lon_vals = np.array(
        fields_group["longitude"], dtype=np.float32
    )  # (num_long,)
    num_lat = grid_lat_vals.shape[0]
    num_lon = grid_lon_vals.shape[0]

    # Construct timestamps
    # Time 0 here is 1959-01-01, 00:00
    print("Constructing timestamps")
    timestamps = fields_group.coords["time"].data.astype("datetime64[s]")

    # Number of seconds since unix time (can be negative)
    seconds_since_epoch = timestamps.astype(np.int64)
    num_time = seconds_since_epoch.shape[0]

    # Create zarr to save to
    forcing_field_shape = (num_time, num_lon, num_lat)
    forcing_fields_dict = {}

    # TOA radiation
    print("Generating TOA radiation")
    toa_array = gc_sr.get_toa_incident_solar_radiation(
        timestamps,
        grid_lat_vals,
        grid_lon_vals,
    )  # (num_time, num_lat, num_lon)
    # Normalize to [0,1]
    toa_min = toa_array.min()
    toa_max = toa_array.max()
    toa_array = (toa_array - toa_min) / (toa_max - toa_min)
    forcing_fields_dict["toa_incident_radiation"] = toa_array.transpose(0, 2, 1)

    # Year progress
    print("Generating day + year progress features")
    year_progress = gc_du.get_year_progress(seconds_since_epoch)
    # (num_time,)
    year_prog_sin, year_prog_cos = progress_to_sin_cos(year_progress)
    forcing_fields_dict["sin_year_progress"] = np.broadcast_to(
        year_prog_sin[:, np.newaxis, np.newaxis], forcing_field_shape
    )
    forcing_fields_dict["cos_year_progress"] = np.broadcast_to(
        year_prog_cos[:, np.newaxis, np.newaxis], forcing_field_shape
    )

    # Day progress
    # Note that this is slightly off as GC only uses a similar modulo calc.
    day_progress = gc_du.get_day_progress(
        seconds_since_epoch, grid_lon_vals
    )  # (num_time, num_lon)
    day_prog_sin, day_prog_cos = progress_to_sin_cos(day_progress)
    forcing_fields_dict["sin_day_progress"] = np.broadcast_to(
        day_prog_sin[:, :, np.newaxis], forcing_field_shape
    )
    forcing_fields_dict["cos_day_progress"] = np.broadcast_to(
        day_prog_cos[:, :, np.newaxis], forcing_field_shape
    )

    # Save as xarray stored with zarr
    print("Saving xarray")
    coord_names = ("time", "longitude", "latitude")
    xa_ds = xa.Dataset(
        {
            var_name: (coord_names, var_vals)
            for var_name, var_vals in forcing_fields_dict.items()
        },
        coords={coord: fields_group.coords[coord] for coord in coord_names},
    )
    xa_da = (
        xa_ds.to_dataarray("forcing_var")
        .transpose("time", "longitude", "latitude", "forcing_var")
        .chunk({"time": 1, "longitude": -1, "latitude": -1, "forcing_var": -1})
    )
    xa_da.to_zarr(forcing_path, mode="w")
    print("Done!")

    if args.plot:
        # (num_vars, num_time, num_lon, num_lat)
        for time_i, timestamp in enumerate(timestamps):
            time_slice = xa_da.isel(time=time_i)  # (num_lon, num_lat, num_vars)

            for var_name in time_slice.coords["forcing_var"].data:
                forcing_field_xa = time_slice.sel(forcing_var=var_name)
                forcing_field = torch.tensor(
                    forcing_field_xa.to_numpy(), dtype=torch.float32
                ).flatten()
                vis.plot_prediction(
                    forcing_field,
                    forcing_field,
                    title=f"{timestamp} UTC, {var_name}",
                )
                plt.show()


if __name__ == "__main__":
    main()
