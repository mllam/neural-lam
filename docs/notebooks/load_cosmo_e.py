# Standard library
import tarfile
from ctypes.wintypes import SHORT
from pathlib import Path

# Third-party
import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from earthkit.data.sources.stream import StreamFieldList

ds = xr.open_zarr("/scratch/mch/cosuna/neural-lam/zarr/cosmo_ml_data.zarr")
zarr_path = "/scratch/mch/sadamov/cosmo_e.zarr"

LATS = ds.lat.values
LONS = ds.lon.values
SHORTNAME_MAP_INV = {
    "2t": "temperature_2m",
    "tp": "precipitation_1hr",
    "10u": "wind_u_10m",
    "10v": "wind_v_10m",
    "msl": "pressure_sea_level",
    "sp": "surface_pressure",
}
starttimes = ds.time.sel(
    time=slice("2019-10-30T00:00:00", "2020-10-29T00:00:00")
)
# Filter for only 00 and 12 UTC times
starttimes = starttimes.where(
    (starttimes.dt.hour == 0) | (starttimes.dt.hour == 12), drop=True
)
starttimes_str = pd.to_datetime(starttimes.values).strftime("%y%m%d%H")
leadtimes = pd.timedelta_range(start="0h", periods=121, freq="1h")
leadtimes_str = [f"{int(lt.total_seconds() / 3600):03d}" for lt in leadtimes]

# Create custom week starts anchored to first starttime
first_starttime = starttimes.values[0]
week_starts = pd.date_range(
    start=first_starttime, end=starttimes.values[-1], freq="7D"
)

# Process each week
for week_start in week_starts:
    week_end = week_start + pd.Timedelta(days=6)
    week_starttimes = starttimes.sel(time=slice(week_start, week_end))
    print(f"Processing week starting {week_start.strftime('%Y-%m-%d')}")

    # Initialize empty dictionaries to store weekly data
    week_data = {var_name: [] for var_name in SHORTNAME_MAP_INV.values()}
    week_times = []
    week_leads = []

    # Loop over start times for this week
    for st, start_time in enumerate(week_starttimes):
        # Determine archive path based on year
        if start_time.dt.year == 2019:
            ARCHIVE_ROOT = Path("/archive/mch/msopr/osm/COSMO-E/FCST19/")
        else:
            ARCHIVE_ROOT = Path("/archive/mch/msopr/osm/COSMO-E/FCST20/")

        start_time_str = pd.to_datetime(start_time.values).strftime("%y%m%d%H")

        # Open tar archive for this start time
        tar_archive = ARCHIVE_ROOT / f"{start_time_str}_205.tar"
        tar_archive = tarfile.open(tar_archive)

        # Loop over lead times
        for lt in leadtimes_str:
            filename = f"{start_time_str}_205/grib/ceffsurf{lt}_000"
            try:
                stream = tar_archive.extractfile(filename)
                streamfieldlist: StreamFieldList = ekd.from_source(
                    "stream", stream
                )

                # Extract fields
                for field in streamfieldlist:
                    shortName = field.metadata("shortName")
                    if shortName in SHORTNAME_MAP_INV:
                        var_name = SHORTNAME_MAP_INV[shortName]
                        week_data[var_name].append(
                            field.values.reshape(LATS.shape)
                        )

                stream.close()
                week_times.append(start_time)
                week_leads.append(pd.Timedelta(hours=int(lt)))

            except KeyError:
                print(f"Missing file for {start_time.values} lead time {lt}")
                continue

        tar_archive.close()

    # Convert lists to numpy arrays
    for var_name in week_data:
        week_data[var_name] = np.stack(week_data[var_name])

    # Convert time coordinates to proper datetime64[ns] format
    week_times_np = np.array(week_starttimes.values, dtype="datetime64[ns]")
    # Create leadtimes array (converting hours to nanoseconds)
    leadtimes_np = np.arange(len(leadtimes)) * 3600 * 1_000_000_000
    leadtimes_np = leadtimes_np.astype("timedelta64[ns]")

    # Create xarray Dataset for this week
    week_ds = xr.Dataset(
        {
            var_name: (
                ["time", "lead_time", "y", "x"],
                week_data[var_name].reshape(
                    len(week_starttimes), len(leadtimes), *LATS.shape
                ),
            )
            for var_name in week_data
        },
        coords={
            "lon": (["y", "x"], LONS),
            "lat": (["y", "x"], LATS),
            "time": week_times_np,
            "lead_time": leadtimes_np,
        },
    )

    # Write to zarr, appending if store exists
    if Path(zarr_path).exists():
        week_ds.to_zarr(zarr_path, mode="a", append_dim="time")
    else:
        week_ds.to_zarr(zarr_path, mode="w")
