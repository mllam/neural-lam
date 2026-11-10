"""Model evaluation and benchmarking suite."""

# Third-party
import torch

# Local
from .evaluator import BenchmarkScorecard, ForecastBenchmark

try:
    torch.serialization.add_safe_globals([BenchmarkScorecard])
except AttributeError:
    pass

__all__ = ["BenchmarkScorecard", "ForecastBenchmark"]
