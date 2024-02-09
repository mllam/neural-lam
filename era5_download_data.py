"""Makes requests to Copernicus Data Store

API request in the format:

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'geopotential',
        'pressure_level': [
            '150', '250',
        ],
        'year': [
            '1964', '1970',
        ],
        'month': [
            '01', '07',
        ],
        'day': [
            '01', '07',
        ],
        'time': [
            '00:00', '06:00',
        ],
    },
    'download.nc')
"""

import cdsapi
import xarray as xr
import zarr
import numcodecs
import sys
from pathlib import Path

DATASET_PATH = "/work/ec249/ec249/bet20/dataset/era5"

c = cdsapi.Client()

region = 'global'
year = [2022]
month = [i for i in range(2, 13)]

year_str = [str(i) for i in year]
month_str = [str(i).zfill(2) for i in month]
day_str = [str(i).zfill(2) for i in range(1, 32)]
time_str = ['00:00', '06:00', '12:00', '18:00']

dir_path = f'{DATASET_PATH}/{region}'
path = Path(dir_path)
path.mkdir(parents=True, exist_ok=True)

# Download month by month
for m in month_str:
    file_path = f'{dir_path}/{year_str[0]}_{m}.nc'
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential', 'specific_humidity','temperature',
                    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                ],
                'pressure_level': [
                    '50', '150', '250', '400', '600', '850', '1000'
                ],
                'year': year_str,
                'month': m,
                'day': day_str,
                'time': time_str,
                #'area': area,
                'format': 'netcdf',
            },
            file_path
        )
    except Exception as e:
        print("=========== Error occured: ===========")
        print(e)
