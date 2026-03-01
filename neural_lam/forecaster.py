import abc
from typing import Callable, Optional, Tuple

# Third-party
import torch
import torch.nn as nn


class Forecaster(nn.Module, abc.ABC):
    """
    Abstract base class for full forecast producers.

    Maps (init_states, forcing, true_states) → (prediction, pred_std) over a
    complete forecast window. Sits between ForecasterModule (Lightning
    orchestration) and StepPredictor (single-step neural network).

    Responsibilities
    ----------------
    * Define the forecasting strategy (AR, direct, ensemble, …).
    * Compute the training loss via compute_loss.
    * Logging, plotting, and optimizer setup belong in ForecasterModule.

    Shape convention
    ----------------
    B         : batch size
    T_init    : conditioning time steps (typically 2)
    pred_steps: forecast steps in this batch
    N         : grid nodes (flat spatial dim)
    d_f       : state features
    d_forcing : forcing features (window already flattened)
    """

    @abc.abstractmethod
    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        true_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Produce a full forecast.

        Parameters
        ----------
        init_states      : (B, T_init, N, d_f)       — standardized
        forcing_features : (B, pred_steps, N, d_forcing)
        true_states      : (B, pred_steps, N, d_f)   — standardized;
                           used for boundary forcing at each step

        Returns
        -------
        prediction : (B, pred_steps, N, d_f) — standardized forecast
        pred_std   : (B, pred_steps, N, d_f) or (d_f,) — std-dev;
                     a constant fallback is returned when the predictor
                     does not produce uncertainty estimates
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()."
        )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        pred_std: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a scalar training loss averaged over batch and pred steps.

        Parameters
        ----------
        prediction : (B, pred_steps, N, d_f)
        target     : (B, pred_steps, N, d_f)
        pred_std   : (B, pred_steps, N, d_f) or (d_f,)
        loss_fn    : metric function from neural_lam.metrics
        mask       : (N,) boolean interior-node mask, or None

        Returns
        -------
        torch.Tensor — scalar loss
        """
        return torch.mean(
            loss_fn(
                prediction,
                target,
                pred_std,
                mask=mask,
                average_grid=True,
                sum_vars=True,
            )
        )
