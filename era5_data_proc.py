import xarray as xr
import zarr
import numcodecs
import constants

import os
import glob
import numpy as np
import torch

PROCESSED_DATA_PATH = "data/era5_uk_reduced/samples/train"
uk_bbox = {
    "lat_max": 63,
    "lat_min": 47,
    "lon_max": 4,
    "lon_min": -10,
}

def uk_subset(data):
    """
    Get UK local subset of data
    
    Dodgy hardcoding of lat/lon values due to wrap around
    """
    # Slice for the longitude from 350 to 360
    subset1 = data.sel(latitude=slice(63, 47), longitude=slice(350, 360))
    # Slice for the longitude from 0 to 4
    subset2 = data.sel(latitude=slice(63, 47), longitude=slice(0, 4))
    # Concatenate the two subsets along the longitude dimension
    uk_subset = xr.concat([subset1, subset2], dim='longitude')
    return uk_subset

def save_dataset_samples(subset=None):
    """
    Convert ERA5 NC files to numpy arrays
    Optionally take a subset of the ERA5 data
    
    Only needs to be run once
    """
    nc_files = glob.glob(f'{constants.DATASET_PATH}/global/*.nc')
    proccessed_dataset_path = "data/era5_uk_reduced/samples/train"
    os.makedirs(proccessed_dataset_path, exist_ok=True)

    for j, filepath in enumerate(nc_files):
        data = xr.open_dataset(filepath)
        if subset:
            data = subset(data)
        
        for i, time in enumerate(data['time'].values):
            time = data['time'].values[i]
            sample = data.sel(time=time)
            array = sample.to_array().values # (n_vars, n_levels, n_lat, n_lon)
            time_py = time.astype('M8[ms]').tolist() # numpy.datetime64 -> datetime.datetime
            date_str = time_py.strftime('%Y%m%d%H%M%S') # datetime.datetime -> str
            
            np.save(f'{proccessed_dataset_path}/{date_str}.npy', array)
            print("Proccessed file: ", date_str)

def create_era5_stats():
    name = "2022_02"
    filepath = f'{constants.DATASET_PATH}/global/{name}.nc'
    data = xr.open_dataset(filepath)
    era5_global_mean = data.mean(dim=("time", "latitude", "longitude")).values
    np.save(f'{constants.DATASET_PATH}/global/{name}_mean.npy', era5_global_mean)
    print("Saved global mean at ", f'{constants.DATASET_PATH}/global/{name}_mean.npy')

def create_era5_grid_features(args):
    pass

def create_xy(subset=None):
    """
    Creates the nwp_xy.npy file for the era5 dataset
    Also creates border mask
    """
    nc_files = glob.glob(f'{constants.DATASET_PATH}/global/*.nc')
    proccessed_dataset_path = "data/era5_uk_reduced/static"
    os.makedirs(proccessed_dataset_path, exist_ok=True)
    
    data = xr.open_dataset(nc_files[0])
    if subset:
        data = subset(data)
    
    latitudes = data.latitude.values
    longitudes = data.longitude.values
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)

    t_lat = torch.from_numpy(latitudes)
    t_lon = torch.from_numpy(longitudes)

    lat_lon_grid = torch.stack(
        torch.meshgrid(t_lat, t_lon, indexing="ij"), dim=-1
    ).permute(2, 1, 0) # (2, lon, lat) or (2, x, y)

    grid_array = lat_lon_grid.numpy()
    np.save(os.path.join(proccessed_dataset_path, "nwp_xy.npy"), grid_array)
    
    # Create border mask
    border_mask = np.zeros(grid_array.shape[1:], dtype=bool)
    np.save(os.path.join(proccessed_dataset_path, "border_mask.npy"), border_mask)
    

if __name__ == "__main__":
    # create_era5_stats()
    # save_dataset_samples(uk_subset)
    create_xy(uk_subset)
