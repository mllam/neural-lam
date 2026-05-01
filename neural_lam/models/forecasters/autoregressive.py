# Standard library
from typing import Optional

# Third-party
import torch

# Local
from ...datastore import BaseDatastore
from ..step_predictors.base import StepPredictor
from .base import Forecaster


class ARForecaster(Forecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to
    unroll a forecast. Makes use of a StepPredictor at each AR step.
    """

    def __init__(self, predictor: StepPredictor, datastore: BaseDatastore):
        super().__init__()
        self.predictor = predictor

        # Register boundary/interior masks on the forecaster, not the predictor
        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

    @property
    def predicts_std(self) -> bool:
        return self.predictor.predicts_std

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Unroll the autoregressive model: at each step ``i`` call
        ``self.predictor`` to produce the next state, then overwrite
        boundary nodes with the true value from ``boundary_states[:, i]``.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, d_f)``. The two initial states
            ``[X_{t-1}, X_t]`` used to seed the rollout. Dims: ``B`` is
            batch size, ``2`` is the time index, ``num_grid_nodes`` is the
            number of spatial nodes, and ``d_f`` is the state feature
            dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_static_f)``. Forcing
            features for each predicted step; ``pred_steps`` defines the
            rollout length.
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_f)``. True state
            values used ONLY to overwrite boundary nodes at each AR step.
            The interior prediction at step ``i`` must not depend on
            ``boundary_states[:, i]`` in any other way.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_f)``. Stacked
            per-step forecasts (with boundary overwritten by the true
            value).
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, d_f)`` when the
            wrapped predictor outputs an std, otherwise ``None`` (in which
            case ``ForecasterModule`` substitutes the constant
            per-variable std).
        """

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            boundary_state = boundary_states[:, i]

            pred_state, pred_std = self.predictor(
                prev_state, prev_prev_state, forcing
            )

            # Overwrite boundary with true state using ARForecaster's mask
            new_state = (
                self.boundary_mask * boundary_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if pred_std is not None:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(prediction_list, dim=1)
        # If predictor outputs std, stack it; otherwise return None so
        # ForecasterModule can substitute the constant per_var_std
        if pred_std_list:
            pred_std = torch.stack(pred_std_list, dim=1)
        else:
            pred_std = None

        return prediction, pred_std
