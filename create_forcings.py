# Standard library
import argparse
import io
import os
import shutil
import zipfile

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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

    # Assuming x and y are your original arrays
    x_padded = np.pad(
        x, ((0, 1), (0, 1)), mode="edge"
    )  # Pad with the edge values
    y_padded = np.pad(y, ((0, 1), (0, 1)), mode="edge")
    mask = np.array(
        [
            [
                (
                    Polygon(
                        [
                            (x_padded[i, j], y_padded[i, j]),
                            (x_padded[i, j + 1], y_padded[i, j + 1]),
                            (x_padded[i + 1, j + 1], y_padded[i + 1, j + 1]),
                            (x_padded[i + 1, j], y_padded[i + 1, j]),
                        ]
                    )
                    .intersection(land_geometries)
                    .area
                    / Polygon(
                        [
                            (x_padded[i, j], y_padded[i, j]),
                            (x_padded[i, j + 1], y_padded[i, j + 1]),
                            (x_padded[i + 1, j + 1], y_padded[i + 1, j + 1]),
                            (x_padded[i + 1, j], y_padded[i + 1, j]),
                        ]
                    ).area
                    if Polygon(
                        [
                            (x_padded[i, j], y_padded[i, j]),
                            (x_padded[i, j + 1], y_padded[i, j + 1]),
                            (x_padded[i + 1, j + 1], y_padded[i + 1, j + 1]),
                            (x_padded[i + 1, j], y_padded[i + 1, j]),
                        ]
                    ).area
                    != 0
                    else 0
                )
                for j in range(x_padded.shape[1] - 1)
            ]
            for i in range(x_padded.shape[0] - 1)
        ]
    )
    shutil.rmtree("./shps")

    mask_xr = xr.DataArray(
        mask, dims=("x", "y"), coords={"x": x[:, 0], "y": y[0, :]}
    )

    return mask_xr


def create_boundary_mask(
    xy_state,
    xy_boundary,
    boundary_thickness,
):
    """Create a boundary mask for the neural LAM model."""
    state_x, state_y = xy_state[0], xy_state[1]
    boundary_x, boundary_y = xy_boundary[0], xy_boundary[1]

    state_y_flat, state_x_flat = state_y.flatten(), state_x.flatten()
    boundary_y_flat, boundary_x_flat = (
        boundary_y.flatten(),
        boundary_x.flatten(),
    )

    mask_flat = cutout_mask(
        state_y_flat, state_x_flat, boundary_y_flat, boundary_x_flat
    )

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
        interior_mask.astype(float), shift=(shift_y, shift_x), order=0
    ).astype(bool)

    # Apply binary dilation
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(
        centered_interior_mask, structure, boundary_thickness
    )

    # Wrap around the dimensions
    dilated_mask = np.roll(dilated_mask, -shift_y, axis=0)
    dilated_mask = np.roll(dilated_mask, -shift_x, axis=1)

    interior_mask = xr.DataArray(
        interior_mask.astype(int),
        dims=("x", "y"),
        coords={"y": boundary_y[0, :], "x": boundary_x[:, 0]},
    )
    boundary_mask = xr.DataArray(
        dilated_mask.astype(int),
        dims=("x", "y"),
        coords={"y": boundary_y[0, :], "x": boundary_x[:, 0]},
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
    parser.add_argument("--boundary_thickness", type=int, default=15)
    parser.add_argument("--overlap", type=bool, default=False)
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
    print(f"Land-sea mask saved to {args.zarr_path}")

    if args.plot:
        # Normalize longitude values to be within [-180, 180]
        if land_sea_mask["x"].max() > 180:
            land_sea_mask["x"] = ((land_sea_mask["x"] + 180) % 360) - 180

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        land_sea_mask.sortby("x").transpose().plot(
            x="x", y="y", ax=ax, cmap="ocean_r", alpha=0.6, add_colorbar=False
        )

        # Add country borders and coastlines
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax.coastlines()

        plt.savefig("land_sea_mask.png")

    interior_mask, boundary_mask = create_boundary_mask(
        xy_state, xy_boundary, args.boundary_thickness
    )
    # mask.to_zarr(args.zarr_path, mode="a")
    print(f"Boundary mask saved to {args.zarr_path}")

    if args.plot:
        interior_mask = interior_mask.where(interior_mask == 1, drop=True)
        boundary_mask = boundary_mask.where(boundary_mask == 1, drop=True)
        interior_mask_regrid = interior_mask.interp_like(boundary_mask)
        interior_mask_regrid = interior_mask_regrid.fillna(0)
        interior_mask_bool_inv = ~interior_mask_regrid.astype(bool)
        boundary_mask_updated = boundary_mask.where(
            interior_mask_bool_inv, drop=True
        )

        # Example parameters for the rotated pole projection
        pole_longitude = 10  # Pole longitude in degrees
        pole_latitude = -45  # Pole latitude in degrees

        # Create a RotatedPole CRS
        rotated_pole_crs = ccrs.RotatedPole(
            pole_longitude=pole_longitude, pole_latitude=pole_latitude
        )

        # Expand x and y into grids that match the shape of your data arrays
        xx_interior, yy_interior = np.meshgrid(
            interior_mask.x.values, interior_mask.y.values
        )
        xx_boundary, yy_boundary = np.meshgrid(
            boundary_mask.x.values, boundary_mask.y.values
        )

        xx_boundary = xx_boundary.T[boundary_mask_updated.values == 1]
        yy_boundary = yy_boundary.T[boundary_mask_updated.values == 1]

        # Flatten the grids to pair each x with its corresponding y
        original_lons_interior = xx_interior.flatten()
        original_lats_interior = yy_interior.flatten()
        original_lons_boundary = xx_boundary.flatten()
        original_lats_boundary = yy_boundary.flatten()

        # Transforming points to the rotated pole coordinate system
        transformed_coords_interior = rotated_pole_crs.transform_points(
            ccrs.PlateCarree(), original_lons_interior, original_lats_interior
        )
        transformed_coords_boundary = rotated_pole_crs.transform_points(
            ccrs.PlateCarree(), original_lons_boundary, original_lats_boundary
        )

        # Extracting transformed coordinates
        transformed_lons_interior = transformed_coords_interior[:, 0]
        transformed_lats_interior = transformed_coords_interior[:, 1]
        transformed_lons_boundary = transformed_coords_boundary[:, 0]
        transformed_lats_boundary = transformed_coords_boundary[:, 1]

        # Plotting
        fig = plt.figure(figsize=(15, 10))
        ax_rotated_pole = fig.add_subplot(
            111,
            projection=ccrs.RotatedPole(
                pole_longitude=pole_longitude, pole_latitude=pole_latitude
            ),
        )

        # Scatter plot of transformed points for both masks
        ax_rotated_pole.scatter(
            transformed_lons_boundary,
            transformed_lats_boundary,
            color="red",
            marker="o",
            s=8,  # Increased size for better visibility
            transform=rotated_pole_crs,
        )
        ax_rotated_pole.scatter(
            transformed_lons_interior,
            transformed_lats_interior,
            color="blue",
            marker="+",
            s=6,  # Increased size for better visibility
            transform=rotated_pole_crs,
        )

        # Invert y-axis if desired
        ax_rotated_pole.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax_rotated_pole.invert_yaxis()

        # Adding coastlines and gridlines
        ax_rotated_pole.coastlines()
        ax_rotated_pole.gridlines()

        plt.savefig("boundary_mask_pointynut.png")


if __name__ == "__main__":
    main()
