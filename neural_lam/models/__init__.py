# Local
from .forecaster_module import ForecasterModule
from .graph_lam import GraphLAM
from .hi_lam import HiLAM
from .hi_lam_parallel import HiLAMParallel

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}
