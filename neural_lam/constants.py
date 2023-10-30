import cartopy.crs as ccrs
import numpy as np

wandb_project = "neural-lam"

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
grid_shape = (582, 390)  # (x, y)

grid_limits = [  # In projection
    -0.6049805,  # min x
    17.48751,  # max x
    42.1798,  # min y
    50.35996,  # max y
]

cosmo_proj = ccrs.PlateCarree()
selected_proj = cosmo_proj
pollon = -170.0
pollat = 43.0

# Plotting
fig_size = (9, 11)
example_file = "data/cosmo/samples/train/laf2015112800_extr.nc"

# Time step prediction during training / prediction (eval)
train_horizon = 3  # hours (t-1 + t -> t+1)
eval_horizon = 25  # hours (autoregressive)

# Log prediction error for these time steps forward
val_step_log_errors = np.arange(1, eval_horizon - 1)
metrics_initialized = False

# Some constants useful for sub-classes
batch_static_feature_dim = 0  # Only open water?
grid_forcing_dim = 0  # 5 features for 3 time-step window
grid_state_dim = len(vertical_levels) * len(param_names)  # 7*4=28
