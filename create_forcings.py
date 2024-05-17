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
from affine import Affine
from anemoi.datasets.grids import cutout_mask
from pyproj import Proj, Transformer
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, center_of_mass


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
            (
                pd.Timestamp(dt_obj)
                - pd.Timestamp(f"{pd.Timestamp(dt_obj).year}-01-01")
            ).total_seconds()
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


def download_and_extract_shapefile(url, tempdir):
    """Download and extract a shapefile from a URL."""
    response = requests.get(url, timeout=10)
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(tempdir)
    return next((f for f in zip_ref.namelist() if f.endswith(".shp")), None)


def generate_land_sea_mask(xy_state, tempdir, projection, high_res_factor=10):
    """Generate a land-sea mask for the neural LAM model."""
    url = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"
    shp_path = download_and_extract_shapefile(url, tempdir)

    gdf = gpd.read_file(os.path.join(tempdir, shp_path))
    gdf = gdf.to_crs(projection)
    gdf = gdf[gdf.is_valid]
    land_geometry = gdf.unary_union

    x, y = xy_state[0], xy_state[1]
    nrows, ncols = y.shape
    xres = (x.max() - x.min()) / float(ncols)
    yres = (y.max() - y.min()) / float(nrows)

    # align center of pixels
    transform = Affine.translation(
        x.min() - xres / 2, y.min() - yres / 2
    ) * Affine.scale(xres, yres)

    high_res_out_shape = (nrows * high_res_factor, ncols * high_res_factor)
    high_res_transform = transform * transform.scale(
        1 / high_res_factor, 1 / high_res_factor
    )

    high_res_out_image = rasterize(
        shapes=[land_geometry.__geo_interface__],
        out_shape=high_res_out_shape,
        transform=high_res_transform,
        fill=0,
        dtype=np.uint8,
    )

    aggregated_out_image = high_res_out_image.reshape(
        (nrows, high_res_factor, ncols, high_res_factor)
    ).mean(axis=(1, 3))

    land_sea_mask = xr.DataArray(
        aggregated_out_image,
        dims=("y", "x"),
        coords={
            "proj_y": (("y", "x"), y),
            "proj_x": (("y", "x"), x),
        },
    )

    return land_sea_mask


def create_boundary_mask(
    lonlat_state,
    lonlat_boundary,
    boundary_thickness,
    overlap=False,
):
    """Create a boundary mask for the neural LAM model."""
    state_lon, state_lat = lonlat_state[0], lonlat_state[1]
    boundary_lon, boundary_lat = lonlat_boundary[0], lonlat_boundary[1]

    state_lat_flat, state_lon_flat = state_lat.flatten(), state_lon.flatten()
    boundary_lat_flat, boundary_lon_flat = (
        boundary_lat.flatten(),
        boundary_lon.flatten(),
    )

    mask_flat = cutout_mask(
        state_lat_flat, state_lon_flat, boundary_lat_flat, boundary_lon_flat
    )

    mask = mask_flat.reshape((boundary_lat.shape)).astype(bool)
    interior_mask = ~mask

    # Find the center of mass of the interior mask
    center_lat, center_lon = center_of_mass(interior_mask)

    # Calculate the shift needed to center the mask
    shift_lat = interior_mask.shape[0] // 2 - int(center_lat)
    shift_lon = interior_mask.shape[1] // 2 - int(center_lon)

    centered_interior_mask = np.roll(
        interior_mask, shift=(shift_lat, shift_lon), axis=(0, 1)
    )

    # Create the boundary mask with appropriate thickness
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(
        centered_interior_mask, structure, boundary_thickness
    )

    dilated_mask = np.roll(dilated_mask, -shift_lat, axis=0)
    dilated_mask = np.roll(dilated_mask, -shift_lon, axis=1)

    if overlap:
        boundary_mask = dilated_mask
    else:
        boundary_mask = dilated_mask & ~interior_mask

    interior_mask = xr.DataArray(
        interior_mask.astype(int),
        dims=("y", "x"),
        coords={
            "lat": (("y", "x"), boundary_lat),
            "lon": (("y", "x"), boundary_lon),
        },
    )
    boundary_mask = xr.DataArray(
        boundary_mask.astype(int),
        dims=("y", "x"),
        coords={
            "lat": (("y", "x"), boundary_lat),
            "lon": (("y", "x"), boundary_lon),
        },
    )
    return interior_mask, boundary_mask


def generate_toa_radiation_forcing(ds, lonlat):
    """
    Pre-compute all static features related to the grid nodes
    """

    # This simplification is only for demonstration purposes
    # Rectangular assumption doesn't hold near pole
    lon, lat = lonlat[0][0, :], lonlat[1][:, 0]

    timestamps = ds.time.values.astype("datetime64[s]")

    toa_array = gc_sr.get_toa_incident_solar_radiation(
        timestamps,
        lat,
        lon,
    )  # (num_time, num_lat, num_lon)
    toa_min = toa_array.min()
    toa_max = toa_array.max()
    toa_array = (toa_array - toa_min) / (toa_max - toa_min)
    toa_radiation = toa_array.transpose(0, 2, 1)

    toa_radiation = xr.DataArray(
        toa_radiation,
        dims=("time", "x", "y"),
        coords={
            "time": ds.time,
            "lat": (("y", "x"), lonlat[1]),
            "lon": (("y", "x"), lonlat[0]),
        },
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
    parser.add_argument("--boundary_thickness", type=int, default=30)
    parser.add_argument("--overlap", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()

    lambert_proj_params = {
        "a": 6367470,
        "b": 6367470,
        "lat_0": 63.3,
        "lat_1": 63.3,
        "lat_2": 63.3,
        "lon_0": 15.0,
        "proj": "lcc",
    }

    lambert_proj = ccrs.LambertConformal(
        central_longitude=lambert_proj_params["lon_0"],
        central_latitude=lambert_proj_params["lat_0"],
        standard_parallels=(
            lambert_proj_params["lat_1"],
            lambert_proj_params["lat_2"],
        ),
    )

    # Make sure the example data is available
    xy_state = np.load("data/meps_example/static/nwp_xy.npy")
    latlong_proj = Proj(proj="latlong", datum="WGS84")
    transformer = Transformer.from_proj(lambert_proj, latlong_proj)
    x_state = xy_state[0]
    y_state = xy_state[1]
    # pylint: disable=unpacking-non-sequence
    lon_state, lat_state = transformer.transform(x_state, y_state)
    lonlat_state = np.stack((lon_state, lat_state), axis=0)

    ds_meps = np.load(
        "data/meps_example/samples/train/nwp_2022040100_mbr000.npy"
    )
    data_vars = {
        f"var_{i}": (("time", "y", "x"), ds_meps[:, :, :, i])
        for i in range(ds_meps.shape[3])
    }
    # This is certainly wrong, but its only for demonstration purposes
    reference_time = np.datetime64("1990-01-01T00:00:00", "ns")
    normalized_hours = np.arange(len(ds_meps[:, 0, 0, 0]))
    hourly_timesteps = reference_time + normalized_hours.astype(
        "timedelta64[h]"
    )
    ds_state = xr.Dataset(
        data_vars,
        coords={
            "time": hourly_timesteps,
        },
    )

    # open era5 from weatherbench
    ds_boundary = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_"
        "10-wb13-6h-1440x721_with_derived_variables.zarr"
    )
    ds_boundary = ds_boundary.sel(time=slice("1990-01-01", "1990-01-02"))
    lon_boundary, lat_boundary = ds_boundary.longitude, ds_boundary.latitude
    (
        llon,
        llat,
    ) = np.meshgrid(lon_boundary, lat_boundary)
    lonlat_boundary = np.stack((llon, llat), axis=0)

    datetime_forcing = calculate_datetime_forcing(ds_state, args)
    datetime_forcing.to_zarr(args.zarr_path, mode="w")
    print(f"Datetime forcing saved to {args.zarr_path}")

    land_sea_mask = generate_land_sea_mask(xy_state, args.tempdir, lambert_proj)
    land_sea_mask.to_zarr(args.zarr_path, mode="w")
    print(f"Land-sea mask saved to {args.zarr_path}")

    interior_mask, boundary_mask = create_boundary_mask(
        lonlat_state, lonlat_boundary, args.boundary_thickness, args.overlap
    )
    boundary_mask.to_zarr(f"boundary_{args.zarr_path}", mode="w")
    print(f"Boundary mask saved to boundary_{args.zarr_path}")

    toa_radiation_state = generate_toa_radiation_forcing(ds_state, lonlat_state)
    toa_radiation_state.to_zarr(args.zarr_path, mode="w")
    toa_radtiation_boundary = generate_toa_radiation_forcing(
        ds_boundary, lonlat_boundary
    )
    toa_radtiation_boundary.to_zarr(f"boundary_{args.zarr_path}", mode="w")
    print(
        f"TOA radiation saved to {args.zarr_path} and boundary_{args.zarr_path}"
    )

    if args.plot:

        _, ax = plt.subplots(2, 2, figsize=(10, 8))

        datetime_forcing.hour_sin.plot(ax=ax[0, 0])
        datetime_forcing.hour_cos.plot(ax=ax[0, 1])
        datetime_forcing.year_sin.plot(ax=ax[1, 0])
        datetime_forcing.year_cos.plot(ax=ax[1, 1])

        plt.tight_layout()
        plt.savefig("datetime_forcing_state.png")

        _, ax = plt.subplots(
            subplot_kw={"projection": lambert_proj}, figsize=(10, 6)
        )
        land_sea_mask.plot(
            x="proj_x",
            y="proj_y",
            ax=ax,
            cmap="viridis",
        )
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax.coastlines()
        ax.gridlines()
        plt.savefig("land_sea_mask.png")

        lons_boundary = boundary_mask.lon.values[boundary_mask == 1].flatten()
        lats_boundary = boundary_mask.lat.values[boundary_mask == 1].flatten()
        lons_interior = interior_mask.lon.values[interior_mask == 1].flatten()
        lats_interior = interior_mask.lat.values[interior_mask == 1].flatten()

        # Transforming points to the projected coordinate system
        transformed_coords_interior = lambert_proj.transform_points(
            ccrs.PlateCarree(), lons_interior, lats_interior
        )
        transformed_coords_boundary = lambert_proj.transform_points(
            ccrs.PlateCarree(), lons_boundary, lats_boundary
        )

        _, ax = plt.subplots(
            subplot_kw={"projection": lambert_proj}, figsize=(10, 6)
        )
        ax.scatter(
            transformed_coords_interior[:, 0],
            transformed_coords_interior[:, 1],
            color="red",
            marker="o",
            s=8,
            transform=lambert_proj,
        )
        ax.scatter(
            transformed_coords_boundary[:, 0],
            transformed_coords_boundary[:, 1],
            color="blue",
            marker="+",
            s=6,
            transform=lambert_proj,
        )
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax.coastlines()
        ax.gridlines()

        plt.savefig("boundary_mask.png")

        _, ax = plt.subplots(figsize=(10, 6))
        toa_radiation_state_mean = toa_radiation_state.mean(dim="y")
        toa_radiation_state_mean.plot(x="x", y="time", ax=ax, cmap="viridis")
        ax.set_xlabel("X")
        ax.set_ylabel("Time")
        ax.set_title("Hovmöller Diagram - TOA Radiation (State)")
        plt.tight_layout()
        plt.savefig("toa_radiation_state_hovmoller.png")


if __name__ == "__main__":
    main()
