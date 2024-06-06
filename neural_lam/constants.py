# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "neural-lam"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 5, 10, 15, 19])
# Also save checkpoints for minimum loss at these lead times
VAL_STEP_CHECKPOINTS = (1, 19)

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = [
    "val_spsk_ratio",
    "val_spread",
]
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    6: [2, 19],  # t_2
    14: [2, 19],  # wvint_0
    15: [2, 19],  # z_1000
}

# Plot forecasts for these variables at given lead times during validation step
# Format is a dictionary that maps from a variable index to a list of
# lead time steps
VAL_PLOT_VARS = {
    4: [2, 19],  # r_2
    14: [2, 19],  # wvint_0
}

# During validation, plot example samples of latent variable from prior and
# variational distribution
LATENT_SAMPLES_PLOT = 4  # Number of samples to plot

# Variable names
PARAM_NAMES = [
    "pres_heightAboveGround_0_instant",
    "pres_heightAboveSea_0_instant",
    "nlwrs_heightAboveGround_0_accum",
    "nswrs_heightAboveGround_0_accum",
    "r_heightAboveGround_2_instant",
    "r_hybrid_65_instant",
    "t_heightAboveGround_2_instant",
    "t_hybrid_65_instant",
    "t_isobaricInhPa_500_instant",
    "t_isobaricInhPa_850_instant",
    "u_hybrid_65_instant",
    "u_isobaricInhPa_850_instant",
    "v_hybrid_65_instant",
    "v_isobaricInhPa_850_instant",
    "wvint_entireAtmosphere_0_instant",
    "z_isobaricInhPa_1000_instant",
    "z_isobaricInhPa_500_instant",
]

PARAM_NAMES_SHORT = [
    "pres_0g",
    "pres_0s",
    "nlwrs_0",
    "nswrs_0",
    "r_2",
    "r_65",
    "t_2",
    "t_65",
    "t_500",
    "t_850",
    "u_65",
    "u_850",
    "v_65",
    "v_850",
    "wvint_0",
    "z_1000",
    "z_500",
]
PARAM_UNITS = [
    "Pa",
    "Pa",
    "W/m²",
    "W/m²",
    "-",  # unitless
    "-",
    "K",
    "K",
    "K",
    "K",
    "m/s",
    "m/s",
    "m/s",
    "m/s",
    "kg/m²",
    "m²/s²",
    "m²/s²",
]

# Projection and grid
# Hard coded for now, but should eventually be part of dataset desc. files
GRID_SHAPE = (268, 238)  # (y, x)

LAMBERT_PROJ_PARAMS = {
    "a": 6367470,
    "b": 6367470,
    "lat_0": 63.3,
    "lat_1": 63.3,
    "lat_2": 63.3,
    "lon_0": 15.0,
    "proj": "lcc",
}

GRID_LIMITS = [  # In projection
    -1059506.5523409774,  # min x
    1310493.4476590226,  # max x
    -1331732.4471934352,  # min y
    1338267.5528065648,  # max y
]

# Create projection
LAMBERT_PROJ = cartopy.crs.LambertConformal(
    central_longitude=LAMBERT_PROJ_PARAMS["lon_0"],
    central_latitude=LAMBERT_PROJ_PARAMS["lat_0"],
    standard_parallels=(
        LAMBERT_PROJ_PARAMS["lat_1"],
        LAMBERT_PROJ_PARAMS["lat_2"],
    ),
)

# Data dimensions
GRID_FORCING_DIM = 5 * 3 + 1  # 5 feat. for 3 time-step window + 1 batch-static
GRID_STATE_DIM = 17
