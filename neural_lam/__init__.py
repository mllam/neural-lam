# Standard library
import importlib.metadata
import os

# Force a non-interactive matplotlib backend by default so that any import
# of neural_lam from a headless or multi-threaded context (HPC training
# jobs, DataLoader workers) does not try to spin up a GUI backend like Tk,
# which raises `RuntimeError: main thread is not in main loop`. Use
# `setdefault` so users running interactively (e.g. Jupyter) keep their
# preferred backend when they set MPLBACKEND themselves before importing
# neural_lam.
os.environ.setdefault("MPLBACKEND", "Agg")

# First-party
import neural_lam.gnn_layers  # noqa: E402
import neural_lam.metrics  # noqa: E402
import neural_lam.models  # noqa: E402
import neural_lam.utils  # noqa: E402
import neural_lam.vis  # noqa: E402

# Local
from .weather_dataset import WeatherDataset  # noqa: E402

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
