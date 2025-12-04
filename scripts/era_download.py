# Download script for ERA5 matching interior domain
# For LAM model project
# by Joel Oskarsson, joel.oskarsson@outlook.com
# adapted by Simon Adamov simon.adamov@meteoswiss.ch

"""Subset ERA5 to interior domain and write Zarr.

Select the time window explicitly with --start/--end
and optionally set lon/lat bounds.
"""

# Standard library
import argparse
import os

# Third-party
import numcodecs
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# ERA5 from WeatherBench2 (public GCS bucket path)
era = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
)


def _get_time_slice_and_out_name(
    start: str | None, end: str | None, output_name: str | None
):
    """Return (time_slice, out_name) using explicit start/end;
    both are required.

    If --output-name isn't provided, defaults to cosmo_era5.zarr.
    """
    if not (start and end):
        raise SystemExit(
            "Both --start and --end must be provided "
            "(e.g. 2016-01-01T00 2016-02-29T18)."
        )
    time_slice = slice(start, end)
    out_name = output_name or "cosmo_era5.zarr"
    return time_slice, out_name


BOUNDARY_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "specific_humidity",
    "surface_pressure",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "land_sea_mask",
    "geopotential_at_surface",
]
SUBSET_PLEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]

# Normalize longitude to [-180, 180] for easier slicing
longitude_new = np.where(
    era["longitude"] > 180, era["longitude"] - 360, era["longitude"]
)
era = era.assign_coords(longitude=longitude_new).sortby("longitude")


def main():
    parser = argparse.ArgumentParser(
        description="Subset ERA5 for interior domain and write Zarr"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.environ.get("WORKDIR", "."),
        help="Directory to write output Zarr "
        "(default: $WORKDIR if set, else current directory)",
    )
    parser.add_argument(
        "--lon-min", type=float, default=-16.0, help="Minimum longitude bound"
    )
    parser.add_argument(
        "--lon-max", type=float, default=33.0, help="Maximum longitude bound"
    )
    parser.add_argument(
        "--lat-min", type=float, default=27.0, help="Minimum latitude bound"
    )
    parser.add_argument(
        "--lat-max", type=float, default=66.0, help="Maximum latitude bound"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start datetime, e.g. 2016-01-01T00",
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End datetime, e.g. 2016-02-29T18"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output Zarr directory name",
    )
    args = parser.parse_args()

    time_slice, out_name = _get_time_slice_and_out_name(
        start=args.start, end=args.end, output_name=args.output_name
    )

    # ERA latitude is typically descending;
    # replicate prior logic by slicing (lat_max -> lat_min)
    era_subset = era.sel(
        longitude=slice(args.lon_min, args.lon_max),
        latitude=slice(args.lat_max, args.lat_min),
        level=SUBSET_PLEVELS,
    )[BOUNDARY_VARS].sel(time=time_slice)

    # Restore longitude to [0, 360) ordering used originally if desired
    longitude_back = (era_subset["longitude"] + 360) % 360
    era_subset = era_subset.assign_coords(longitude=longitude_back).sortby(
        "longitude"
    )

    print("ERA subset summary:")
    print(era_subset)
    print(f"Uncompressed size: {era_subset.nbytes / 1e9:.2f} GB")

    compressor = numcodecs.Blosc(
        cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
    )
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    print(f"Writing Zarr -> {out_path}")
    with ProgressBar():
        era_subset.to_zarr(
            out_path,
            encoding={
                var: {"compressor": compressor} for var in era_subset.data_vars
            },
            zarr_format=2,
        )


if __name__ == "__main__":
    main()
