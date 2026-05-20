# Standard library
import importlib.metadata

# First-party
import neural_lam.gnn_layers
import neural_lam.metrics
import neural_lam.models
import neural_lam.utils
import neural_lam.vis

# Local
from .weather_dataset import WeatherDataset

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
