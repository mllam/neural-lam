"""Neural-LAM: graph-based neural weather prediction models."""

# Standard library
import importlib.metadata
import os

# Third-party
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

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
