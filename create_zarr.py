import os

import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


def process_grib_files_in_folder(folder_path, filter_by_keys, idx_folder):
    """Process all grib2 files in a given folder with specified filter."""
    grib_files = [os.path.join(folder_path, f)
                  for f in os.listdir(folder_path)]
    datasets = []
    for file_path in grib_files:
        # Generate a unique index file path for each GRIB file
        idx_path = os.path.join(idx_folder, os.path.basename(file_path) + '.idx')
        ds = xr.open_dataset(
            file_path,
            engine='cfgrib',
            backend_kwargs={
                'filter_by_keys': filter_by_keys,
                'indexpath': idx_path})
        datasets.append(ds)
    return datasets


def main(data_in, data_out_2015_2019, data_out_2020, idx_folder):
    all_datasets_2015_2019 = []
    all_datasets_2020 = []

    for root, dirs, files in os.walk(data_in):
        if 'det' in dirs:
            det_path = os.path.join(root, 'det')
            print(f"Processing {det_path} for surface")
            surface_datasets = process_grib_files_in_folder(
                det_path, {'typeOfLevel': 'surface'}, idx_folder)
            print(f"Processing {det_path} for heightAboveGround")
            height_datasets = process_grib_files_in_folder(
                det_path, {'typeOfLevel': 'heightAboveGround'}, idx_folder)

            combined_datasets = surface_datasets + height_datasets
            for ds in combined_datasets:
                year = int(ds.time.dt.year[0])
                if year == 2020:
                    all_datasets_2020.append(ds)
                elif 2015 <= year <= 2019:
                    all_datasets_2015_2019.append(ds)

    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

    if all_datasets_2015_2019:
        combined_ds_2015_2019 = xr.concat(all_datasets_2015_2019, dim='time')
        with ProgressBar():
            combined_ds_2015_2019.to_zarr(
                data_out_2015_2019, consolidated=True, mode='w',
                encoding={var: {'compressor': compressor}
                          for var in combined_ds_2015_2019.variables})

    if all_datasets_2020:
        combined_ds_2020 = xr.concat(all_datasets_2020, dim='time')
        with ProgressBar():
            combined_ds_2020.to_zarr(
                data_out_2020, consolidated=True, mode='w',
                encoding={var: {'compressor': compressor}
                          for var in combined_ds_2020.variables})


if __name__ == "__main__":
    data_in = '/scratch/mch/dealmeih/kenda/'
    data_out_2015_2019 = '/scratch/mch/sadamov/output_2015_2019.zarr'
    data_out_2020 = '/scratch/mch/sadamov/output_2020.zarr'
    idx_folder = '/path/to/custom/idx_folder'  # Specify your custom path here
    main(data_in, data_out_2015_2019, data_out_2020, idx_folder)
