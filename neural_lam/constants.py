# Third-party
import numpy as np
from cartopy import crs as ccrs

WANDB_PROJECT = "neural-lam"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

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

FORCING_NAMES_SHORT = ["ASHFL_S", "ASOB_S", "ATHB_S"]
FORCING_NAMES = [
    "Surface sensible heat flux",
    "Surface solar radiation",
    "Surface thermal radiation",
]


# Variable names ordered and extended based on the PARAM_NAMES_SHORT list
PARAM_NAMES = [
    "Pressure Deviation",
    "Specific humidity",
    "Relative humidity",
    "Temperature",
    "Zonal wind component",
    "Meridional wind component",
    "Vertical velocity",
    "Cloud cover total",
    "Pressure at Mean Sea Level",
    "Surface Pressure",
    "2-meter Temperature",
    "Total Precipitation",
    "10-meter Zonal wind speed",
    "10-meter Meridional wind speed",
]


# Short names
PARAM_NAMES_SHORT = [
    "PP",
    "QV",
    "RELHUM",
    "T",
    "U",
    "V",
    "W",
    "CLCT",
    "PMSL",
    "PS",
    "T_2M",
    "TOT_PREC",
    "U_10M",
    "V_10M",
]

# Units
PARAM_UNITS = [
    "Pa",
    "kg/kg",
    "%",
    "K",
    "m/s",
    "m/s",
    "Pa/s",
    "%",
    "Pa",
    "Pa",
    "K",
    "kg/m^2",
    "m/s",
    "m/s",
]

# Parameter weights
PARAM_WEIGHTS = {
    "PP": 1,
    "QV": 1,
    "RELHUM": 1,
    "T": 1,
    "U": 1,
    "V": 1,
    "W": 1,
    "CLCT": 1,
    "PMSL": 1,
    "PS": 1,
    "T_2M": 1,
    "TOT_PREC": 1,
    "U_10M": 1,
    "V_10M": 1,
}

# Vertical levels
VERTICAL_LEVELS = [
    1,
    6,
    9,
    12,
    14,
    16,
    20,
    23,
    27,
    31,
    39,
    45,
    60,
]

PARAM_CONSTRAINTS = {
    "RELHUM": (0, 100),
    "CLCT": (0, 100),
    # "TQV": (0, None),
    "TOT_PREC": (0, None),
}

IS_3D = {
    "PP": 1,
    "QV": 1,
    "RELHUM": 1,
    "T": 1,
    "U": 1,
    "V": 1,
    "W": 1,
    "CLCT": 0,
    "PMSL": 0,
    "PS": 0,
    "T_2M": 0,
    "TOT_PREC": 0,
    "U_10M": 0,
    "V_10M": 0,
}

# Vertical level weights
# These were retrieved based on the pressure levels of
# https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5
LEVEL_WEIGHTS = {
    1: 1,
    6: 1,
    9: 1,
    12: 1,
    14: 1,
    16: 1,
    20: 1,
    23: 1,
    27: 1,
    31: 1,
    39: 1,
    45: 1,
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
EXAMPLE_FILE = "data/cosmo/samples/train/data.zarr"
EVAL_DATETIMES = ["2020050400"]  # prev_prev timestep (t-2)
EVAL_PLOT_VARS = ["T_2M"]
STORE_EXAMPLE_DATA = True
SELECTED_PROJ = ccrs.PlateCarree()
SMOOTH_BOUNDARIES = False

# Some constants useful for sub-classes 3 fluxes variables + 4 time-related
# features; in packages of three (prev, prev_prev, current)
GRID_FORCING_DIM = (3 + 4) * 3
GRID_STATE_DIM = sum(
    len(VERTICAL_LEVELS) if IS_3D[param] else 1 for param in PARAM_NAMES_SHORT
)
