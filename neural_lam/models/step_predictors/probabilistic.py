"""Interface for latent-variable step predictors."""

# Standard library
from abc import abstractmethod
from typing import Optional, Tuple

# Third-party
import torch

# Local
from .base import StepPredictor


class LatentStepPredictor(StepPredictor):
    """
    Abstract step predictor for latent-variable models.

    Extends :class:`StepPredictor` for predictors that advance the state by
    sampling a latent variable and decoding it. Subclasses implement
    :meth:`step_distributions`, which returns the latent prior, the
    variational posterior and the decoded reconstruction for one time step.
    The evaluation :meth:`forward` is provided here as the prior-sampling
    special case. The predictor produces only distributions and a
    reconstruction; likelihood, KL and the ELBO are assembled outside it.
    """

    @abstractmethod
    def step_distributions(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        compute_prior: bool = True,
    ) -> Tuple[
        Optional[torch.distributions.Distribution],
        Optional[torch.distributions.Distribution],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Latent distributions and decoded reconstruction for one time step.

        The latent sample is drawn from the variational posterior when a
        ``target`` is given (training path) and from the prior otherwise
        (evaluation path). This is the only entry point that takes the
        target. The predictor computes no likelihood or KL.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, num_forcing_vars)``. External forcings
            for this step.
        target : torch.Tensor or None, optional
            Shape ``(B, num_grid_nodes, num_state_vars)``. The true next state
            ``X_{t+1}``. When given, the latent is sampled from the
            variational posterior conditioned on it; when ``None``, from the
            prior.
        compute_prior : bool, optional
            If ``False``, skip the prior on the training path (``prior_dist``
            is returned as ``None``); used when the KL weight is zero. On the
            evaluation path the prior is always computed.

        Returns
        -------
        prior_dist : torch.distributions.Distribution or None
            Prior over the latent variable, or ``None`` when skipped.
        posterior_dist : torch.distributions.Distribution or None
            Variational posterior over the latent variable, or ``None`` on the
            evaluation path.
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. Decoded mean given
            the latent sample.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, num_state_vars)`` when ``output_std``
            is True, otherwise ``None``.
        """

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Advance one step by sampling the latent prior and decoding.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, num_forcing_vars)``. External forcings
            for this step.

        Returns
        -------
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. Predicted next state
            ``X_{t+1}``.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, num_state_vars)`` when ``output_std``
            is True, otherwise ``None``.
        """
        _, _, pred_mean, pred_std = self.step_distributions(
            prev_state, prev_prev_state, forcing, target=None
        )
        return pred_mean, pred_std
