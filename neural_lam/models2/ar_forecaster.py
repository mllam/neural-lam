# Third-party
import torch

# Local
from .forecaster import Forecaster
from .step_predictor import StepPredictor


class ARForecaster(Forecaster):
    """
    Auto-regressive forecaster that unrolls a StepPredictor in time.
    """

    def __init__(self, step_predictor: StepPredictor, args=None):
        super().__init__()
        self.step_predictor = step_predictor
        self.args = args

    def _prepare_true_states(
        self,
        true_states: torch.Tensor,
        batch_size: int,
        pred_steps: int,
        ensemble_size: int,
    ) -> torch.Tensor:
        """
        Prepare boundary states for AR unroll.

        Accepts either:
        - (B, T, N, d_f)
        - (B, S, T, N, d_f)
        """
        if true_states.dim() == 4:
            if true_states.shape[0] != batch_size:
                raise ValueError(
                    "true_states batch size must match init_states batch size"
                )
            if true_states.shape[1] < pred_steps:
                raise ValueError(
                    "true_states time dimension is smaller than pred_steps"
                )
            true_states = true_states[:, :pred_steps]
            if ensemble_size > 1:
                true_states = true_states.repeat_interleave(
                    ensemble_size, dim=0
                )
            return true_states

        if true_states.dim() == 5:
            if true_states.shape[0] != batch_size:
                raise ValueError(
                    "true_states batch size must match init_states batch size"
                )
            if true_states.shape[1] != ensemble_size:
                raise ValueError(
                    "true_states ensemble dimension must match ensemble_size"
                )
            if true_states.shape[2] < pred_steps:
                raise ValueError(
                    "true_states time dimension is smaller than pred_steps"
                )
            true_states = true_states[:, :, :pred_steps]
            return true_states.reshape(
                batch_size * ensemble_size,
                pred_steps,
                *true_states.shape[3:],
            )

        raise ValueError(
            "true_states must have shape (B,T,N,d_f) or (B,S,T,N,d_f)"
        )

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        true_states: torch.Tensor = None,
        pred_steps: int = None,
        ensemble_size: int = 1,
    ):
        """
        Roll out autoregressive prediction.

        init_states: (B, 2, N, d_f)
        forcing_features: (B, T, N, d_forcing)
        true_states: (B, T, N, d_f) or (B, S, T, N, d_f)
        """
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")

        batch_size = init_states.shape[0]
        max_steps = forcing_features.shape[1]
        if pred_steps is None:
            pred_steps = max_steps
        if pred_steps > max_steps:
            raise ValueError("pred_steps cannot exceed forcing time dimension")

        # Trim forcing if caller requested a shorter unroll.
        forcing_features = forcing_features[:, :pred_steps]

        # Expand batch for ensemble execution.
        if ensemble_size > 1:
            init_states = init_states.repeat_interleave(ensemble_size, dim=0)
            forcing_features = forcing_features.repeat_interleave(
                ensemble_size, dim=0
            )

        if true_states is not None:
            true_states = self._prepare_true_states(
                true_states=true_states,
                batch_size=batch_size,
                pred_steps=pred_steps,
                ensemble_size=ensemble_size,
            )

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []

        for step_idx in range(pred_steps):
            forcing = forcing_features[:, step_idx]

            pred_state, pred_std = self.step_predictor.predict_step(
                prev_state, prev_prev_state, forcing
            )

            # Match ARModel behavior: overwrite boundary with truth when given.
            if true_states is not None:
                border_state = true_states[:, step_idx]
                new_state = (
                    self.step_predictor.boundary_mask * border_state
                    + self.step_predictor.interior_mask * pred_state
                )
            else:
                new_state = pred_state

            prediction_list.append(new_state)
            if self.step_predictor.output_std:
                pred_std_list.append(pred_std)

            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(prediction_list, dim=1)
        if self.step_predictor.output_std:
            pred_std = torch.stack(pred_std_list, dim=1)
        else:
            # Keep ARModel contract: use constant per-variable std if model
            # does not predict std directly.
            pred_std = self.step_predictor.per_var_std

        if ensemble_size > 1:
            prediction = prediction.reshape(
                batch_size, ensemble_size, pred_steps, *prediction.shape[2:]
            )
            if self.step_predictor.output_std:
                pred_std = pred_std.reshape(
                    batch_size, ensemble_size, pred_steps, *pred_std.shape[2:]
                )

        return prediction, pred_std
