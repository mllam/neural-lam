# Third-party
import xarray as xr

data_urls = [
    "https://mllam-test-data.s3.eu-north-1.amazonaws.com/single_levels.zarr",
    "https://mllam-test-data.s3.eu-north-1.amazonaws.com/height_levels.zarr",
]

local_paths = [
    "data/danra/single_levels.zarr",
    "data/danra/height_levels.zarr",
]

for url, path in zip(data_urls, local_paths):
    print(f"Downloading {url} to {path}")
    ds = xr.open_zarr(url)
    chunk_dict = {dim: -1 for dim in ds.dims if dim != "time"}
    chunk_dict["time"] = 20
    ds = ds.chunk(chunk_dict)

    for var in ds.variables:
        if "chunks" in ds[var].encoding:
            del ds[var].encoding["chunks"]

    ds.to_zarr(path, mode="w")
    print("DONE")
