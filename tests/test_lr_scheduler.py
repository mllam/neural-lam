# Standard library
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


def _make_model_args(graph_name):
    " \\Create a minimal ModelArgs for testing.\\\
