import abc
from typing import Optional, Tuple

# Third-party
import torch
import torch.nn as nn


class StepPredictor(nn.Module, abc.ABC):
    """
    Abstract base class for single-step state predictors.

    Maps (X_{t-1}, X_t, F_t) → X_{t+1}, corresponding to f̂ in Oskarsson
    et al.  Subclasses implement the neural network (GNN, CNN, ViT, …).

    Responsibilities
    ----------------
    * Neural network forward pass only.
    * Normalization, boundary masking, and AR unrolling belong in
      ARForecaster, not here.

    Shape convention (symbols used throughout)
    ------------------------------------------
    B        : batch size
    N        : number of grid nodes (flat spatial dim)
    d_f      : number of state features
    d_forcing: forcing features (window already flattened into last dim)
    """

    # Set to True in subclasses that also output a predicted std-dev.
    output_std: bool = False

    @abc.abstractmethod
    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict the next state.

        Parameters
        ----------
        prev_state      : (B, N, d_f)       — X_t, standardized
        prev_prev_state : (B, N, d_f)       — X_{t-1}, standardized
        forcing         : (B, N, d_forcing) — forcing for this step

        Returns
        -------
        new_state : (B, N, d_f)       — predicted X_{t+1}, standardized
        pred_std  : (B, N, d_f) or None — aleatoric std, None if
                                          output_std is False
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()."
        )
