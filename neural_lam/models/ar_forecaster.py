# Third-party
import torch

# Local
from .forecaster import Forecaster
from .step_predictor import StepPredictor


class ARForecaster(Forecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to
    unroll a forecast. Makes use of a StepPredictor at each AR step.
    """

    def __init__(self, predictor: StepPredictor):
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        border_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unroll the autoregressive model.
        border_states is used ONLY to overwrite boundary nodes at each step.
        The interior prediction at step i must not depend on
        border_states[:, i] in any other way.
        """

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = border_states[:, i]

            pred_state, pred_std = self.predictor(
                prev_state, prev_prev_state, forcing
            )

            # Overwrite border with true state using predictor's mask
            new_state = (
                self.predictor.boundary_mask * border_state
                + self.predictor.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.predictor.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(prediction_list, dim=1)
        if self.predictor.output_std:
            pred_std = torch.stack(pred_std_list, dim=1)
        else:
            pred_std = self.predictor.per_var_std

        return prediction, pred_std
