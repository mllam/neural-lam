# Third-party
import numpy as np
from cartopy import crs as ccrs

WANDB_PROJECT = "neural-lam"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 5, 10, 15, 19])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = []
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    6: [2, 19],  # t_2
    14: [2, 19],  # wvint_0
    15: [2, 19],  # z_1000
}

# Variable names
PARAM_NAMES = [
    "Temperature",
    "Zonal wind component",
    "Meridional wind component",
    "Relative humidity",
    "Pressure at Mean Sea Level",
    "Pressure Perturbation",
    "Surface Pressure",
    "Total Precipitation",
    "Total Water Vapor content",
    "2-meter Temperature",
    "10-meter Zonal wind speed",
    "10-meter Meridional wind speed",
]

# Short names
PARAM_NAMES_SHORT = [
    "T",
    "U",
    "V",
    "RELHUM",
    "PMSL",
    "PP",
    "PS",
    "TOT_PREC",
    "TQV",
    "T_2M",
    "U_10M",
    "V_10M",
]

# Units
PARAM_UNITS = [
    "K",
    "m/s",
    "m/s",
    "Perc.",
    "Pa",
    "hPa",
    "Pa",
    "$kg/m^2$",
    "$kg/m^2$",
    "K",
    "m/s",
    "m/s",
]

# Parameter weights
PARAM_WEIGHTS = {
    "T": 1,
    "U": 1,
    "V": 1,
    "RELHUM": 1,
    "PMSL": 1,
    "PP": 1,
    "PS": 1,
    "TOT_PREC": 1,
    "TQV": 1,
    "T_2M": 1,
    "U_10M": 1,
    "V_10M": 1,
}

# Vertical levels
VERTICAL_LEVELS = [1, 5, 13, 22, 38, 41, 60]

PARAM_CONSTRAINTS = {
    "RELHUM": (0, 100),
    "TQV": (0, None),
    "TOT_PREC": (0, None),
}

IS_3D = {
    "T": 1,
    "U": 1,
    "V": 1,
    "RELHUM": 1,
    "PMSL": 0,
    "PP": 1,
    "PS": 0,
    "TOT_PREC": 0,
    "TQV": 0,
    "T_2M": 0,
    "U_10M": 0,
    "V_10M": 0,
}

# Vertical level weights
LEVEL_WEIGHTS = {
    1: 1,
    5: 1,
    13: 1,
    22: 1,
    38: 1,
    41: 1,
    60: 1,
}

# Projection and grid
GRID_SHAPE = (390, 582)  # (y, x)

# Time step prediction during training / prediction (eval)
TRAIN_HORIZON = 3  # hours (t-1 + t -> t+1)
EVAL_HORIZON = 25  # hours (autoregressive)

# Properties of the Graph / Mesh
GRAPH_NUM_CHILDREN = 3

# Log prediction error for these time steps forward
VAL_STEP_LOG_ERRORS = np.arange(1, EVAL_HORIZON - 1)
METRICS_INITIALIZED = False

# Plotting
FIG_SIZE = (15, 10)
EXAMPLE_FILE = "data/cosmo/samples/train/data_2015112800.zarr"
CHUNK_SIZE = 100
EVAL_DATETIME = "2020100215"
EVAL_PLOT_VARS = ["TQV"]
STORE_EXAMPLE_DATA = False
COSMO_PROJ = ccrs.PlateCarree()
SELECTED_PROJ = COSMO_PROJ
POLLON = -170.0
POLLAT = 43.0
SMOOTH_BOUNDARIES = False

# Some constants useful for sub-classes
GRID_FORCING_DIM = 0
GRID_STATE_DIM = sum(
    len(VERTICAL_LEVELS) if IS_3D[param] else 1 for param in PARAM_NAMES_SHORT
)
