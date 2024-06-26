import xarray as xr


credentials_key = "546V9NGV07UQCBM80Y47"
credentials_secret = "8n61wiWFojIkxJM4MC5luoZNBDoitIqvHLXkXs9i"
credentials_endpoint_url = "https://object-store.os-api.cci1.ecmwf.int"

ds = xr.open_zarr(
    "s3://danra/v0.4.0/single_levels.zarr/",
    consolidated=True,
    storage_options={
        "key": credentials_key,
        "secret": credentials_secret,
        "client_kwargs": {"endpoint_url": credentials_endpoint_url},
    },
)
print(ds)