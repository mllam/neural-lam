"""
Lightning modules wrapping forecasters for training and evaluation.
"""

# Local
from .base import BaseForecasterModule
from .deterministic import DeterministicForecasterModule
from .probabilistic import ProbabilisticForecasterModule
