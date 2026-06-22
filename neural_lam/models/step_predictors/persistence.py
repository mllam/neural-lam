"""Persistence baseline step predictor."""

# Third-party
import torch

# Local
from ...datastore import BaseDatastore
from .base import StepPredictor


class PersistencePredictor(StepPredictor):
    """
    Trivial baseline predictor that returns the previous state unchanged.

    At each AR step the predicted next state is simply the current state,
    i.e. ``X_{t+1} = X_t``.  This provides a persistence (climatological
    no-change) baseline that can be evaluated through the standard
    ``ForecasterModule`` / ``ARForecaster`` pipeline.
    """

    trainable: bool = False
    """Persistence predictors have no learnable parameters."""

    def __init__(
        self,
        datastore: BaseDatastore,
        output_std: bool = False,
        output_clamping_lower: dict[str, float] | None = None,
        output_clamping_upper: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the PersistencePredictor.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing grid metadata and data access.
        output_std : bool, default False
            Ignored — persistence never predicts uncertainty.
        output_clamping_lower : dict, optional
            Ignored — persistence returns the raw previous state.
        output_clamping_upper : dict, optional
            Ignored — persistence returns the raw previous state.
        **kwargs
            Absorbed so that the standard CLI kwargs (``graph_name``,
            ``hidden_dim``, etc.) do not cause errors.
        """
        super().__init__(
            datastore=datastore,
            output_std=False,
            output_clamping_lower=None,
            output_clamping_upper=None,
        )

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """
        Return the previous state as the prediction.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``.
            The current state ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``.
            The previous state ``X_{t-1}`` (unused).
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, num_forcing_vars)``.
            External forcings (unused).

        Returns
        -------
        pred_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``.
            Equal to ``prev_state``.
        pred_std : None
            Always ``None`` — persistence does not predict uncertainty.
        """
        return prev_state, None
