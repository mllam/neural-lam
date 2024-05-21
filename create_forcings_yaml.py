# Standard library
import argparse
import io
import os
import zipfile

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import graphcast.solar_radiation as gc_sr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from anemoi.datasets.grids import cutout_mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation, shift

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


def generate_land_sea_mask(xy, tempdir, high_res_factor=10):
    """Generate a land-sea mask for the neural LAM model."""

    def download_and_extract_shapefile(url, tempdir):
        response = requests.get(url, timeout=10)
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(tempdir)
        return next((f for f in zip_ref.namelist() if f.endswith(".shp")), None)

    url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
    shp_path = download_and_extract_shapefile(url, tempdir)

    gdf = gpd.read_file(os.path.join(tempdir, shp_path))
    land_geometry = gdf.unary_union

    minx, miny = xy[0][:, 0].min(), xy[1][0, :].min()
    maxx, maxy = xy[0][:, 0].max(), xy[1][0, :].max()

    # Generate a high-resolution binary mask
    high_res_out_shape = (
        int(xy[1].shape[1] * high_res_factor),
        int(xy[0].shape[0] * high_res_factor),
    )
    high_res_transform = from_bounds(
        minx, maxy, maxx, miny, high_res_out_shape[1], high_res_out_shape[0]
    )

    high_res_out_image = rasterize(
        shapes=[land_geometry.__geo_interface__],
        out_shape=high_res_out_shape,
        transform=high_res_transform,
        fill=0,
        dtype=np.uint8,
    )

    # Aggregate the high-resolution mask to the target grid resolution
    aggregated_out_image = high_res_out_image.reshape(
        (xy[1].shape[1], high_res_factor, xy[0].shape[0], high_res_factor)
    ).mean(axis=(1, 3))

    mask_xr = xr.DataArray(
        aggregated_out_image,
        dims=("y", "x"),
        coords={"y": xy[1][0, :], "x": xy[0][:, 0]},
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

    center_y, center_x = np.array(interior_mask.shape) // 2
    shift_y = center_y - (y_min + y_max) // 2
    shift_x = center_x - (x_min + x_max) // 2
    centered_interior_mask = shift(
        interior_mask.astype(float), shift=(shift_y, shift_x), order=0
    ).astype(bool)

    # Create the boundary mask with appropriate thickness
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(
        centered_interior_mask, structure, boundary_thickness
    )

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


def generate_toa_radiation_forcing(ds, xy):
    """
    Pre-compute all static features related to the grid nodes
    """

    x, y = xy[0][:, 0], xy[1][0, :]

    # Time 0 here is 1959-01-01, 00:00
    timestamps = ds.time.values.astype("datetime64[s]")

    toa_array = gc_sr.get_toa_incident_solar_radiation(
        timestamps,
        y,
        x,
    )  # (num_time, num_lat, num_lon)
    toa_min = toa_array.min()
    toa_max = toa_array.max()
    toa_array = (toa_array - toa_min) / (toa_max - toa_min)
    toa_radiation = toa_array.transpose(0, 2, 1)

    toa_radiation = xr.DataArray(
        toa_radiation,
        dims=("time", "x", "y"),
        coords={"time": ds.time, "x": x, "y": y},
    )
    return toa_radiation


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
    ds_boundary = config_loader.open_zarr("boundary")
    xy_state = config_loader.get_nwp_xy("state")
    xy_boundary = config_loader.get_nwp_xy("boundary")

    # reduce the time dim for this example
    ds_state = ds_state.isel(time=slice(0, 24))
    ds_boundary = ds_boundary.isel(time=slice(0, 24))

    datetime_forcing = calculate_datetime_forcing(ds_state, args)
    datetime_forcing.to_zarr(args.zarr_path, mode="w")
    print(f"Datetime forcing saved to {args.zarr_path}")

    land_sea_mask = generate_land_sea_mask(xy_state, args.tempdir)
    land_sea_mask.to_zarr(args.zarr_path, mode="w")
    print(f"Land-sea mask saved to {args.zarr_path}")

    interior_mask, boundary_mask = create_boundary_mask(
        xy_state, xy_boundary, args.boundary_thickness
    )
    boundary_mask.to_zarr(f"boundary_{args.zarr_path}", mode="w")
    print(f"Boundary mask saved to boundary_{args.zarr_path}")

    toa_radiation_state = generate_toa_radiation_forcing(ds_state, xy_state)
    toa_radiation_state.to_zarr(args.zarr_path, mode="w")
    toa_radtiation_boundary = generate_toa_radiation_forcing(
        ds_boundary, xy_boundary
    )
    toa_radtiation_boundary.to_zarr(f"boundary_{args.zarr_path}", mode="w")
    print(f"TOA radiation saved to boundary_{args.zarr_path}")

    if args.plot:

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        datetime_forcing.hour_sin.plot(ax=axs[0, 0])
        datetime_forcing.hour_cos.plot(ax=axs[0, 1])
        datetime_forcing.year_sin.plot(ax=axs[1, 0])
        datetime_forcing.year_cos.plot(ax=axs[1, 1])

        plt.tight_layout()
        plt.savefig("datetime_forcing_state.png")

        # Normalize longitude values to be within [-180, 180]
        if land_sea_mask["x"].max() > 180:
            land_sea_mask["x"] = ((land_sea_mask["x"] + 180) % 360) - 180

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        land_sea_mask.plot(
            x="x", y="y", ax=ax, cmap="ocean_r", alpha=0.6, add_colorbar=False
        )
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax.coastlines()

        plt.savefig("land_sea_mask.png")

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

        xx_interior, yy_interior = np.meshgrid(
            interior_mask.x.values, interior_mask.y.values
        )
        xx_boundary, yy_boundary = np.meshgrid(
            boundary_mask.x.values, boundary_mask.y.values
        )

        xx_boundary = xx_boundary.T[boundary_mask_updated.values == 1]
        yy_boundary = yy_boundary.T[boundary_mask_updated.values == 1]

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

        transformed_lons_interior = transformed_coords_interior[:, 0]
        transformed_lats_interior = transformed_coords_interior[:, 1]
        transformed_lons_boundary = transformed_coords_boundary[:, 0]
        transformed_lats_boundary = transformed_coords_boundary[:, 1]

        fig = plt.figure(figsize=(15, 10))
        ax_rotated_pole = fig.add_subplot(
            111,
            projection=ccrs.RotatedPole(
                pole_longitude=pole_longitude, pole_latitude=pole_latitude
            ),
        )

        ax_rotated_pole.scatter(
            transformed_lons_boundary,
            transformed_lats_boundary,
            color="red",
            marker="o",
            s=8,
            transform=rotated_pole_crs,
        )
        ax_rotated_pole.scatter(
            transformed_lons_interior,
            transformed_lats_interior,
            color="blue",
            marker="+",
            s=6,
            transform=rotated_pole_crs,
        )
        ax_rotated_pole.invert_yaxis()

        ax_rotated_pole.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax_rotated_pole.coastlines()
        ax_rotated_pole.gridlines()

        plt.savefig("boundary_mask.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        toa_radiation_state_mean = toa_radiation_state.mean(dim="x")
        toa_radiation_state_mean.plot(x="y", y="time", ax=ax, cmap="viridis")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Time")
        ax.set_title("Hovmöller Diagram - TOA Radiation (State)")
        plt.tight_layout()
        plt.savefig("toa_radiation_state_hovmoller.png")


if __name__ == "__main__":
    main()
