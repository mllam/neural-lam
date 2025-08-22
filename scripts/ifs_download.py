# Download script for ifs5 matching DANRA domain
# For LAM model project
# by Joel Oskarsson, joel.oskarsson@outlook.com
# adapted by Simon Adamov simon.adamov@meteoswiss.ch

import numcodecs
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# IFS from weatherbench 2
ifs = xr.open_zarr(
    "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
)

# Precomputed limits (adapted to COSMO)
boundary_lon_range = (-16, 33)
boundary_lat_range = (27, 66)

time_slice = slice("2019-08-30T00", "2020-11-02T00")

boundary_vars = [
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
]
subset_plevels = [
    100,
    200,
    400,
    600,
    700,
    850,
    925,
    1000,
]

# Make new longitude coord for slicing
# Create a new longitude coordinate that goes from -180 to 180
longitude_new = np.where(
    ifs["longitude"] > 180, ifs["longitude"] - 360, ifs["longitude"]
)
# Assign the new longitude coordinate to the dataset
ifs = ifs.assign_coords(longitude=longitude_new)
# Sort the dataset by the new longitude coordinate
ifs = ifs.sortby(["longitude", "latitude"])

# Slice
ifs_subset = ifs.sel(
    longitude=slice(*boundary_lon_range),
    latitude=slice(*boundary_lat_range),
    time=time_slice,
    level=subset_plevels,
)[boundary_vars]

# Change back longitude
longitude_back = (ifs_subset["longitude"] + 360) % 360
ifs_subset = ifs_subset.assign_coords(longitude=longitude_back)
ifs_subset = ifs_subset.sortby("longitude").sortby("latitude", ascending=False)

print("ifs Subset:")
print(ifs_subset)
print(f"ifs subset has size {ifs_subset.nbytes / 1e9} GB uncompressed")

# Set up compression
compressor = numcodecs.Blosc(
    cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
)
print("Downloading and saving zarr...")
with ProgressBar():
    ifs_subset.to_zarr(
        "ifs_subset.zarr",
        encoding={
            var: {"compressor": compressor} for var in ifs_subset.data_vars
        },
    )
