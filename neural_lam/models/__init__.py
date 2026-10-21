"""Neural-LAM model architectures including GraphLAM, HiLAM, and variants."""

# Standard library
from typing import Callable, TypeVar

# Local
from .forecasters.autoregressive import ARForecaster
from .forecasters.base import Forecaster
from .module import ForecasterModule
from .step_predictors.base import StepPredictor
from .step_predictors.graph.base import BaseGraphModel
from .step_predictors.graph.graph_efm import GraphEFM, GraphEFMMultiScale
from .step_predictors.graph.graph_lam import GraphLAM
from .step_predictors.graph.hi_lam import HiLAM
from .step_predictors.graph.hi_lam_parallel import HiLAMParallel
from .step_predictors.graph.hierarchical import BaseHiGraphModel
from .step_predictors.persistence import PersistencePredictor

T = TypeVar("T", bound=type[StepPredictor])

MODELS: dict[str, type[StepPredictor]] = {}


def register_model(name: str) -> Callable[[T], T]:
    """
    Decorator to register a StepPredictor class in the MODELS registry.

    Parameters
    ----------
    name : str
        The identifier used to select the model via ``--model``.

    Returns
    -------
    Callable
        Decorator registering the class under ``name``.
    """

    def decorator(cls: T) -> T:
        """Register the model class."""
        if name in MODELS:
            raise ValueError(
                f"Model '{name}' is already registered to {MODELS[name]}"
            )
        MODELS[name] = cls
        return cls

    return decorator


register_model("graph_lam")(GraphLAM)
register_model("hi_lam")(HiLAM)
register_model("hi_lam_parallel")(HiLAMParallel)
register_model("persistence")(PersistencePredictor)
register_model("graph_efm")(GraphEFM)
register_model("graph_efm_multiscale")(GraphEFMMultiScale)
