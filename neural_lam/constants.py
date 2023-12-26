import numpy as np
from cartopy import crs as ccrs

wandb_project = "neural-lam"

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
    1, 5, 13, 22, 38, 41, 60
]


# Vertical level weights
level_weights = {level: 1 for level in vertical_levels}

# Projection and grid
grid_shape = (390, 582)  # (y, x)

# Time step prediction during training / prediction (eval)
train_horizon = 6  # hours (t-1 + t -> t+1)
eval_horizon = 25  # hours (autoregressive)

# Log prediction error for these time steps forward
val_step_log_errors = np.arange(1, eval_horizon - 1)
metrics_initialized = False

# Plotting
fig_size = (15, 10)
example_file = "data/cosmo/samples/train/data_2015112800.zarr"
chunk_size = 100
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
