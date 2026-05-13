# Local
from .forecasters.autoregressive import ARForecaster
from .forecasters.base import Forecaster
from .module import ForecasterModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}
