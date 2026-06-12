"""Neural-LAM model architectures including GraphLAM, HiLAM, and variants."""

# Local
from .forecasters.autoregressive import ARForecaster
from .forecasters.base import Forecaster
from .module import ForecasterModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_efm import GraphEFM
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel

# NOTE: GraphEFM is intentionally NOT registered in MODELS yet. The shared
# construction call in train_model.py instantiates the chosen model with a
# fixed deterministic kwarg set -- datastore-first, no ``config``, and with
# ``mesh_aggr`` -- whereas GraphEFM requires ``config`` (for its per_var_std
# weighting) and takes no ``mesh_aggr``. Registering it requires config-aware
# model assembly in train_model.py.
MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}
