# Create simple static land-sea mask for COSMO dataset
# For LAM model project
# by Simon Adamov, simon.adamov@meteoswiss.ch

# Standard library
import argparse
import io
import os
import zipfile
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import requests
import xarray as xr
from affine import Affine
from rasterio.features import rasterize

pyproj.datadir.get_data_dir()  # This will initialize proj properly


def download_and_extract_shapefile(url, tempdir):
    """Download and extract a shapefile from a URL."""
    response = requests.get(url, timeout=10)
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(tempdir)
    return next((f for f in zip_ref.namelist() if f.endswith(".shp")), None)


def generate_land_sea_mask(lat, lon, tempdir, projection, high_res_factor=10):
    """Generate a land-sea mask for the neural LAM model."""
    url = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"
    shp_path = download_and_extract_shapefile(url, tempdir)

    gdf = gpd.read_file(os.path.join(tempdir, shp_path))
    gdf = gdf.to_crs(projection)
    land_geometry = gdf[gdf.is_valid].union_all()

    # Transform lat/lon to projected coordinates
    transformer = projection.transform_points(ccrs.PlateCarree(), lon, lat)
    x, y = transformer[..., 0], transformer[..., 1]

    # Get dimensions from the input arrays, respecting y, x order
    ny, nx = lat.shape  # y=390, x=582

    xres = (x.max() - x.min()) / float(nx)
    yres = (y.max() - y.min()) / float(ny)

    transform = Affine.translation(
        x.min() - xres / 2, y.min() - yres / 2
    ) * Affine.scale(xres, yres)
    high_res_transform = transform * transform.scale(
        1 / high_res_factor, 1 / high_res_factor
    )

    # Output shape should maintain y, x order
    high_res_out_shape = (ny * high_res_factor, nx * high_res_factor)
    high_res_out_image = rasterize(
        shapes=[land_geometry.__geo_interface__],
        out_shape=high_res_out_shape,
        transform=high_res_transform,
        fill=0,
        dtype=np.uint8,
    )

    # Maintain y, x order in reshaping
    aggregated_out_image = high_res_out_image.reshape(
        (
            ny,
            high_res_factor,
            nx,
            high_res_factor,
        )
    ).mean(axis=(1, 3))

    # Create DataArray with consistent y, x dimensions
    return xr.DataArray(
        aggregated_out_image.T,
        name="lsm",
        dims=("x", "y"),
        coords={"lat": (("x", "y"), lat.T), "lon": (("x", "y"), lon.T)},
        attrs={
            "long_name": "Land-sea mask",
            "description": "Binary mask where 1: land and 0: sea",
            "units": "1",
        },
    ).chunk(None)


def plot_land_sea_mask(zarr_path, output_filename):
    """Plot the land-sea mask from a Zarr archive."""
    # Load the land-sea mask
    ds = xr.open_zarr(zarr_path)
    lsm = ds["lsm"]

    # Define the RotatedPole projection
    lambert_proj = ccrs.RotatedPole(
        pole_longitude=190,
        pole_latitude=43,
    )

    # Create a plot
    fig, ax = plt.subplots(
        subplot_kw={"projection": lambert_proj}, figsize=(10, 6)
    )
    lsm.plot(
        x="lon", y="lat", ax=ax, transform=ccrs.PlateCarree(), cmap="viridis"
    )
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Save the plot
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")


def main():
    """Create and save land-sea mask.

    CLI:
        --source_zarr <path> (path to COSMO zarr; default: cosmo_ml_data.zarr)
        --output_zarr <path> (output zarr; default: cosmo_land_sea_mask.zarr)
    """

    parser = argparse.ArgumentParser(description="Generate land-sea mask")
    parser.add_argument(
        "--source_zarr",
        help="Path to COSMO sample zarr",
        default="cosmo_ml_data.zarr",
    )
    parser.add_argument(
        "--output_zarr",
        help="Output zarr file (default: cosmo_land_sea_mask.zarr)",
        default="cosmo_land_sea_mask.zarr",
    )
    args = parser.parse_args()

    # Get conda environment path
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        proj_lib = str(Path(conda_prefix) / "share" / "proj")
        os.environ["PROJ_LIB"] = proj_lib
        pyproj.datadir.set_data_dir(proj_lib)
        print(f"PROJ_LIB set to {proj_lib}")

    tempdir = "./shps"
    zarr_path = args.output_zarr
    os.makedirs(tempdir, exist_ok=True)

    # Lambert projection parameters for Northern Europe
    lambert_proj = ccrs.RotatedPole(
        pole_longitude=190,
        pole_latitude=43,
    )

    # Load coordinates from zarr archive
    ds = xr.open_zarr(args.source_zarr)
    lat = ds.lat.values
    lon = ds.lon.values

    land_sea_mask = generate_land_sea_mask(lat, lon, tempdir, lambert_proj)
    land_sea_mask.to_zarr(zarr_path, mode="w", zarr_format=2)
    print(f"Land-sea mask saved to {zarr_path} (var='lsm')")

    # Use the new plotting function
    plot_land_sea_mask(
        zarr_path, Path(zarr_path).parent / "cosmo_land_sea_mask_plot.png"
    )


if __name__ == "__main__":
    main()
