import cartopy
import numpy as np

wandb_project = "neural-lam"

seconds_in_year = 365*24*60*60 # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
val_step_log_errors = np.array([1, 2, 3, 5, 10, 15, 19])

# Variable names
param_names = [
    'pres_heightAboveGround_0_instant',
    'pres_heightAboveSea_0_instant',
    'nlwrs_heightAboveGround_0_accum',
    'nswrs_heightAboveGround_0_accum',
    'r_heightAboveGround_2_instant',
    'r_hybrid_65_instant',
    't_heightAboveGround_2_instant',
    't_hybrid_65_instant',
    't_isobaricInhPa_500_instant',
    't_isobaricInhPa_850_instant',
    'u_hybrid_65_instant',
    'u_isobaricInhPa_850_instant',
    'v_hybrid_65_instant',
    'v_isobaricInhPa_850_instant',
    'wvint_entireAtmosphere_0_instant',
    'z_isobaricInhPa_1000_instant',
    'z_isobaricInhPa_500_instant'
]

param_names_short = [
    'pres_0g',
    'pres_0s',
    'nlwrs_0',
    'nswrs_0',
    'r_2',
    'r_65',
    't_2',
    't_65',
    't_500',
    't_850',
    'u_65',
    'u_850',
    'v_65',
    'v_850',
    'wvint_0',
    'z_1000',
    'z_500'
]
param_units = [
    'Pa',
    'Pa',
    'W/m\\textsuperscript{2}',
    'W/m\\textsuperscript{2}',
    '-', # unitless
    '-',
    'K',
    'K',
    'K',
    'K',
    'm/s',
    'm/s',
    'm/s',
    'm/s',
    'kg/m\\textsuperscript{2}',
    'm\\textsuperscript{2}/s\\textsuperscript{2}',
    'm\\textsuperscript{2}/s\\textsuperscript{2}'
]

# Projection and grid
# TODO Do not hard code this, make part of static dataset files
grid_shape = (268, 238) # (y, x)

lambert_proj_params = {
     'a': 6367470,
     'b': 6367470,
     'lat_0': 63.3,
     'lat_1': 63.3,
     'lat_2': 63.3,
     'lon_0': 15.0,
     'proj': 'lcc'
 }

grid_limits = [ # In projection
    -1059506.5523409774, # min x
    1310493.4476590226, # max x
    -1331732.4471934352, # min y
    1338267.5528065648, # max y
]

# Create projection
lambert_proj = cartopy.crs.LambertConformal(
        central_longitude=lambert_proj_params['lon_0'],
        central_latitude=lambert_proj_params['lat_0'],
        standard_parallels=(lambert_proj_params['lat_1'],
        lambert_proj_params['lat_2']))
