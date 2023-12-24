import numcodecs
import numpy as np
from cartopy import crs as ccrs

wandb_project = "neural-lam"

data_config = {
    "data_path": "/scratch/mch/sadamov/ml_v1/",
    "filename_regex": "(.*)_extr.nc",
    "zarr_path": "/users/sadamov/pyprojects/neural-cosmo/data/cosmo/samples",
    "compressor": numcodecs.Blosc(
        cname='lz4',
        clevel=7,
        shuffle=numcodecs.Blosc.SHUFFLE),
    "chunk_size": 100,
    "test_year": 2020,
}

# TODO: fix for leap years
# Assuming no leap years in dataset (2024 is next)
seconds_in_year = 365 * 24 * 60 * 60

# Full names
param_names = [
    'Temperature',
    'Zonal wind component',
    'Meridional wind component',
    'Relative humidity',
]
# Short names
param_names_short = [
    'T',
    'U',
    'V',
    'RELHUM',
]

# Units
param_units = [
    'K',
    'm/s',
    'm/s',
    'Perc.',
]

# Parameter weights
param_weights = {
    'T': 1,
    'U': 1,
    'V': 1,
    'RELHUM': 1,
}

# Vertical levels
vertical_levels = [
    1,
    5,
    13,
    22,
    38,
    41,
    60
]

# Vertical level weights
level_weights = {
    1: 1,
    5: 1,
    13: 1,
    22: 1,
    38: 1,
    41: 1,
    60: 1
}

# Projection and grid
grid_shape = (390, 582)  # (y, x)

# Zoom for graph plotting
zoom_limit = 1e10

# Time step prediction during training / prediction (eval)
train_horizon = 3  # hours (t-1 + t -> t+1)
eval_horizon = 25  # hours (autoregressive)

# Log prediction error for these time steps forward
val_step_log_errors = np.arange(1, eval_horizon - 1)
metrics_initialized = False

# Plotting
fig_size = (15, 10)
example_file = "data/cosmo/samples/train/laf2015112800_extr.nc"
eval_sample = 0
store_example_data = False
cosmo_proj = ccrs.PlateCarree()
selected_proj = cosmo_proj
pollon = -170.0
pollat = 43.0

# Some constants useful for sub-classes
batch_static_feature_dim = 0
grid_forcing_dim = 0
grid_state_dim = len(vertical_levels) * len(param_names)
