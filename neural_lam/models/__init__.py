"""Neural-LAM model architectures including GraphLAM, HiLAM, and variants."""

# Local
from .forecasters.autoregressive import unroll_forecast
from .forecasters.base import BaseForecaster
from .forecasters.deterministic import (
    BaseDeterministicForecaster,
    DeterministicARForecaster,
)
from .forecasters.graph_efm import GraphEFMForecaster
from .forecasters.probabilistic import (
    BaseProbabilisticARForecaster,
    BaseProbabilisticForecaster,
)
from .modules.base import BaseForecastingModule
from .modules.deterministic import DeterministicForecastingModule
from .modules.probabilistic import ProbabilisticForecastingModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_efm import GraphEFM, GraphEFMMultiScale
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel
from .step_predictors.persistence import PersistencePredictor

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
    "persistence": PersistencePredictor,
    "graph_efm": GraphEFM,
    "graph_efm_ms": GraphEFMMultiScale,
}
