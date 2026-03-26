# Third-party
from torch import nn


class Forecaster(nn.Module):
    """
    Base class for full-horizon forecasting modules.
    """

    def forward(
        self,
        init_states,
        forcing_features,
        true_states=None,
        pred_steps=None,
        ensemble_size=1,
    ):
        """
        Build a full forecast horizon.
        """
        raise NotImplementedError("Subclasses must implement forward")
