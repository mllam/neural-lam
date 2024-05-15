# Standard library
import argparse
import io
import os
import shutil
import zipfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Third-party
import numpy as np
import pandas as pd
import requests
import shapefile
import xarray as xr
from anemoi.datasets.grids import cutout_mask
from scipy.ndimage import binary_dilation, shift
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

# First-party
from neural_lam.utils import ConfigLoader


def calculate_datetime_forcing(ds, args):
    """Calcuye the datetime forcing for the neural LAM model."""
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


def generate_land_sea_mask(xy, tempdir):
    """Generate a land-sea mask for the neural LAM model."""
    x, y = xy[0], xy[1]
    url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
    extracted_files = download_natural_earth_data(url, tempdir)
    shp_path = next((f for f in extracted_files if f.endswith(".shp")), None)

    sf = shapefile.Reader(os.path.join(tempdir, shp_path))
    land_geometries = unary_union(
        [shape(s) for s in sf.shapes() if s.shapeType == shapefile.POLYGON]
    )

    # Calcuye the proportion of each grid cell covered by land
    mask = np.array(
        [
            [
                Polygon(
                    [
                        (x[i, j], y[i, j]),
                        (x[i, j + 1], y[i, j]),
                        (x[i + 1, j + 1], y[i + 1, j]),
                        (x[i + 1, j], y[i + 1, j]),
                    ]
                )
                .intersection(land_geometries)
                .area
                / Polygon(
                    [
                        (x[i, j], y[i, j]),
                        (x[i, j + 1], y[i, j]),
                        (x[i + 1, j + 1], y[i + 1, j]),
                        (x[i + 1, j], y[i + 1, j]),
                    ]
                ).area
                for j in range(x.shape[1] - 1)
            ]
            for i in range(y.shape[0] - 1)
        ]
    )

    shutil.rmtree("./shps")
    mask = xr.Dataset({"land_sea_mask": (("y", "x"), mask)})

    return mask


def create_boundary_mask(xy_state, xy_boundary, boundary_thickness, overlap):
    state_x, state_y = xy_state[0], xy_state[1]
    boundary_x, boundary_y = xy_boundary[0], xy_boundary[1]

    state_y_flat, state_x_flat = state_y.flatten(), state_x.flatten()
    boundary_y_flat, boundary_x_flat = boundary_y.flatten(), boundary_x.flatten()

    mask_flat = cutout_mask(
        state_y_flat,
        state_x_flat,
        boundary_y_flat,
        boundary_x_flat)

    mask = mask_flat.reshape((boundary_y.shape)).astype(bool)
    interior_mask = ~mask

    # Find the bounding box of the interior mask
    coords = np.argwhere(interior_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Calculate the center of the mask
    center_y, center_x = np.array(interior_mask.shape) // 2

    # Calculate the shift needed to center the interior mask
    shift_y = center_y - (y_min + y_max) // 2
    shift_x = center_x - (x_min + x_max) // 2

    # Shift the padded interior mask to the center
    centered_interior_mask = shift(
        interior_mask.astype(float),
        shift=(shift_y, shift_x),
        order=0).astype(bool)

    # Apply binary dilation
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(
        centered_interior_mask,
        structure,
        boundary_thickness)

    # Wrap around the dimensions
    dilated_mask = np.roll(dilated_mask, -shift_y, axis=0)
    dilated_mask = np.roll(dilated_mask, -shift_x, axis=1)

    if overlap:
        boundary_mask = dilated_mask
    else:
        boundary_mask = dilated_mask & ~interior_mask

    interior_mask = xr.DataArray(
        interior_mask.astype(int), dims=("x", "y"),
        coords={
            "y": boundary_y[0, :],
            "x": boundary_x[:, 0]
        }
    )
    boundary_mask = xr.DataArray(
        boundary_mask.astype(int), dims=("x", "y"),
        coords={
            "y": boundary_y[0, :],
            "x": boundary_x[:, 0]
        }
    )
    return interior_mask, boundary_mask


def main():
    """Main function for creating the datetime forcing and boundary mask."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config", type=str, default="neural_lam/data_config.yaml"
    )
    parser.add_argument("--zarr_path", type=str, default="forcings.zarr")
    parser.add_argument("--tempdir", type=str, default="./shps")
    parser.add_argument("--seconds_in_year", type=int, default=31536000)
    parser.add_argument("--boundary_thickness", type=int, default=40)
    parser.add_argument("--overlap", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()

    config_loader = ConfigLoader(args.data_config)
    ds_state = config_loader.open_zarr("state")
    xy_state = config_loader.get_nwp_xy("state")
    xy_boundary = config_loader.get_nwp_xy("boundary")

    datetime_forcing = calculate_datetime_forcing(ds_state, args)
    datetime_forcing.to_zarr(args.zarr_path, mode="a")
    print(f"Datetime forcing saved to {args.zarr_path}")

    land_sea_mask = generate_land_sea_mask(xy_state, args.tempdir)
    land_sea_mask.to_zarr(args.zarr_path, mode="a")

    interior_mask, boundary_mask = create_boundary_mask(
        xy_state, xy_boundary, args.boundary_thickness, args.overlap
    )
    # mask.to_zarr(args.zarr_path, mode="a")
    print(f"Boundary mask saved to {args.zarr_path}")

    if args.plot:
        interior_mask = interior_mask.where(interior_mask == 1, drop=True)
        boundary_mask = boundary_mask.where(boundary_mask == 1, drop=True)

        # Normalize longitude values to be within [-180, 180]
        if boundary_mask['x'].max() > 180:
            boundary_mask['x'] = ((boundary_mask['x'] + 180) % 360) - 180
        if interior_mask['x'].max() > 180:
            interior_mask['x'] = ((interior_mask['x'] + 180) % 360) - 180

    # Assuming interior_mask is your xarray DataArray
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the boundary mask normally
    boundary_mask.sortby("x").transpose().plot(
        x='x',
        y='y',
        ax=ax,
        cmap='Reds',
        alpha=0.5,
        add_colorbar=False)

    # Create a patch for the hatched area
    patch = mpatches.Rectangle(
        (interior_mask.x.min(), interior_mask.y.min()),
        width=interior_mask.x.max() - interior_mask.x.min(),
        height=interior_mask.y.max() - interior_mask.y.min(),
        edgecolor="blue", hatch='//', fill=False, linewidth=2,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(patch)

    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.coastlines()

    plt.savefig("boundary_mask.png")


if __name__ == "__main__":
    main()
