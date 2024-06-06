# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "neural-lam"

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 5, 10, 20, 40])
# Also save checkpoints for minimum loss at these lead times
VAL_STEP_CHECKPOINTS = np.array([1, 20, 40])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = [
    "val_spsk_ratio",
    "val_rmse",
]
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    7: [1, 10],  # z500
    78: [1, 10],  # 2t
    79: [1, 10],  # 10u
}

# Plot forecasts for these variables at given lead times during validation step
# Format is a dictionary that maps from a variable index to a list of
# lead time steps
VAL_PLOT_VARS = {
    7: np.array([2, 20]),  # z500
    22: np.array([2, 20]),  # q700
    36: np.array([2, 20]),  # t850
    78: np.array([2, 20]),  # 2t
    79: np.array([2, 20]),  # 10u
    80: np.array([2, 20]),  # 10v
    82: np.array([2, 20]),  # tp
}

# During validation, plot example samples of latent variable from prior and
# variational distribution
LATENT_SAMPLES_PLOT = 4  # Number of samples to plot

# Following table 2 in GC
# Keys to read from fields zarr
ATMOSPHERIC_PARAMS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]  # times 13 pressure levels = 78 params

SURFACE_PARAMS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]  # = 5 params
# Total = 83 params

# Variable names
ATMOSPHERIC_PARAMS_SHORT = [
    "z",
    "q",
    "t",
    "u",
    "v",
    "w",
]
SURFACE_PARAMS_SHORT = ["2t", "10u", "10v", "msl", "tp"]
PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]  # 13 levels
PARAM_NAMES_SHORT = [
    f"{param}{level}"
    for param in ATMOSPHERIC_PARAMS_SHORT
    for level in PRESSURE_LEVELS
] + SURFACE_PARAMS_SHORT

ATMOSPHERIC_PARAMS_UNITS = [
    "m²/s²",
    "kg/kg",
    "K",
    "m/s",
    "m/s",
    "Pa/s",
]
PARAM_UNITS = [
    unit for unit in ATMOSPHERIC_PARAMS_UNITS for level in PRESSURE_LEVELS
] + ["K", "m/s", "m/s", "Pa", "m"]

# What variables (index) to plot during evaluation

EVAL_PLOT_VARS = np.concatenate(
    [
        level_start_i
        + np.arange(0, len(ATMOSPHERIC_PARAMS)) * len(PRESSURE_LEVELS)
        for level_start_i in (
            PRESSURE_LEVELS.index(level) for level in (200, 500, 850)
        )
    ]
    + [np.arange(78, 83)]  # Surface
)

# Projection and grid
GRID_SHAPE = (240, 121)  # (long, lat)

# Create projection
MAP_PROJ = cartopy.crs.Robinson()
GRID_LIMITS = [
    -0.75,
    359.25,
    -90,
    90,
]

# Time step length (hours)
TIME_STEP_LENGTH = 6

# Data dimensions
GRID_ORIGINAL_FORCING_DIM = 5  # 5 features
GRID_FORCING_DIM = GRID_ORIGINAL_FORCING_DIM * 3
# 5 features for 3 time-step window
GRID_STATE_DIM = 6 * 13 + 5  # 83
