"""Neural-LAM model architectures including GraphLAM, HiLAM, and variants."""

# Local
from .forecasters.autoregressive import ARForecaster
from .forecasters.base import Forecaster
from .forecasters.graph_efm import GraphEFMForecaster
from .forecasters.probabilistic import (
    ProbabilisticARForecaster,
    ProbabilisticForecaster,
)
from .modules.base import BaseForecasterModule
from .modules.deterministic import DeterministicForecasterModule
from .modules.probabilistic import ProbabilisticForecasterModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_efm import GraphEFM, GraphEFMMultiScale
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel

# Graph-EFM models are probabilistic: train_model.py builds them with a
# config-aware, probabilistic assembly path (GraphEFMForecaster wrapped in a
# ProbabilisticForecasterModule), distinct from the deterministic models
# above. ``PROBABILISTIC_MODELS`` marks which entries take that path.
MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
    "graph_efm": GraphEFM,
    "graph_efm_ms": GraphEFMMultiScale,
}

PROBABILISTIC_MODELS = {"graph_efm", "graph_efm_ms"}
