"""Neural-LAM model architectures including GraphLAM, HiLAM, and variants."""

# Local
from .forecasters.autoregressive import unroll_forecast
from .forecasters.base import BaseForecaster
from .forecasters.deterministic import (
    BaseDeterministicForecaster,
    DeterministicARForecaster,
)
from .forecasters.ensemble import (
    BaseEnsembleARForecaster,
    BaseEnsembleForecaster,
)
from .modules.base import BaseForecastingModule
from .modules.deterministic import DeterministicForecastingModule
from .modules.ensemble import EnsembleForecastingModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_efm import GraphEFM, GraphEFMMultiScale
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel

# NOTE: GraphEFM/GraphEFMMultiScale are intentionally NOT registered in
# MODELS yet.
# ``train_model.build_predictor`` instantiates the chosen model with a fixed
# deterministic kwarg set -- datastore-first, no ``config``, and with
# ``mesh_aggr`` -- whereas the Graph-EFM models require ``config`` (for their
# per_var_std weighting) and take no ``mesh_aggr``.
# Registering them requires making ``build_predictor`` config-aware.
MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}
