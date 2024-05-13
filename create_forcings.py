# Standard library
import argparse
import io
import os
import shutil
import zipfile

# Third-party
import numpy as np
import pandas as pd
import requests
import shapefile
import xarray as xr
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

# First-party
from neural_lam.utils import ConfigLoader


def calculate_datetime_forcing(ds, args):
    """Calculate the datetime forcing for the neural LAM model."""
    time_dim = ds.time
    dt_range = pd.date_range(
        start=pd.to_datetime(time_dim[0].values).to_pydatetime(),
        end=pd.to_datetime(time_dim[-1].values).to_pydatetime(),
        freq="h",
        name="time",
    )
    dt_range_xr = xr.DataArray(dt_range, dims=["time"])
    hours_of_day = xr.DataArray(dt_range_xr.dt.hour, dims=["time"])
    seconds_into_year = xr.DataArray(
        [
            (pd.Timestamp(dt_obj).year - pd.Timestamp(dt_obj).year)
            * args.seconds_in_year
            for dt_obj in dt_range_xr.values
        ],
        dims=["time"],
    )
    hour_angle = (hours_of_day / 12) * np.pi
    year_angle = (seconds_into_year / args.seconds_in_year) * 2 * np.pi
    datetime_forcing = xr.Dataset(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "year_sin": np.sin(year_angle),
            "year_cos": np.cos(year_angle),
        },
        coords={"time": dt_range_xr},
    )
    datetime_forcing = (datetime_forcing + 1) / 2
    return datetime_forcing


def download_natural_earth_data(url, tempdir):
    """Download and extract Natural Earth data."""
    response = requests.get(url, timeout=10)
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(tempdir)
        extracted_files = [f for f in zip_ref.namelist() if f.endswith(".shp")]
        print(f"Extracted files: {extracted_files}")
        return extracted_files


def generate_land_sea_mask(ds, tempdir):
    """Generate a land-sea mask for the neural LAM model."""
    lat = ds.lat.values
    lon = ds.lon.values
    url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
    extracted_files = download_natural_earth_data(url, tempdir)
    shp_path = next((f for f in extracted_files if f.endswith(".shp")), None)

    sf = shapefile.Reader(os.path.join(tempdir, shp_path))
    land_geometries = unary_union(
        [shape(s) for s in sf.shapes() if s.shapeType == shapefile.POLYGON]
    )

    # Calculate the proportion of each grid cell covered by land
    mask = np.array(
        [
            [
                Polygon(
                    [
                        (lon[i, j], lat[i, j]),
                        (lon[i, j + 1], lat[i, j]),
                        (lon[i + 1, j + 1], lat[i + 1, j]),
                        (lon[i + 1, j], lat[i + 1, j]),
                    ]
                )
                .intersection(land_geometries)
                .area
                / Polygon(
                    [
                        (lon[i, j], lat[i, j]),
                        (lon[i, j + 1], lat[i, j]),
                        (lon[i + 1, j + 1], lat[i + 1, j]),
                        (lon[i + 1, j], lat[i + 1, j]),
                    ]
                ).area
                for j in range(lon.shape[1] - 1)
            ]
            for i in range(lat.shape[0] - 1)
        ]
    )

    shutil.rmtree("./shps")
    return mask


def main():
    """Main function for creating the datetime forcing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config", type=str, default="neural_lam/data_config.yaml"
    )
    parser.add_argument("--zarr_path", type=str, default="forcings.zarr")
    parser.add_argument("--tempdir", type=str, default="./shps")
    parser.add_argument("--seconds_in_year", type=int, default=31536000)
    args = parser.parse_args()
    config_loader = ConfigLoader(args.data_config)
    ds = config_loader.open_zarr("state")
    datetime_forcing = calculate_datetime_forcing(ds, args)

    # land_sea_mask = generate_land_sea_mask(ds, args.tempdir)

    datetime_forcing.to_zarr(args.zarr_path, mode="w")
    print(f"Datetime forcing saved to {args.zarr_path}")


if __name__ == "__main__":
    main()
