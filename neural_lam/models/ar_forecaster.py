# Third-party
import torch

# Local
from ..datastore import BaseDatastore
from .forecaster import Forecaster
from .step_predictor import StepPredictor


class ARForecaster(Forecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to
    unroll a forecast. Makes use of a StepPredictor at each AR step.
    """

    def __init__(self, predictor: StepPredictor, datastore: BaseDatastore):
        super().__init__()
        self.predictor = predictor

        # Register boundary/interior masks here, not in StepPredictor (Item 7)
        boundary_mask = (
            torch.tensor(
                datastore.boundary_mask.values, dtype=torch.float32
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unroll the autoregressive model.
        boundary_states is used ONLY to overwrite boundary nodes at each step.
        The interior prediction at step i must not depend on
        boundary_states[:, i] in any other way.
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
        # ForecasterModule can substitute the constant per_var_std (Item 5)
        if pred_std_list:
            pred_std = torch.stack(pred_std_list, dim=1)
        else:
            pred_std = None

        return prediction, pred_std
