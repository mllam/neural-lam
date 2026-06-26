"""Autoregressive forecaster for Graph-EFM latent-variable predictors."""

# Standard library
from typing import Optional

# Third-party
import torch

# Local
from ..step_predictors.probabilistic import LatentStepPredictor
from .autoregressive import ARForecaster


class ProbabilisticARForecaster(ARForecaster):
    """
    Autoregressive forecaster for latent-variable step predictors.

    The inherited :meth:`ARForecaster.forward` is the evaluation rollout: each
    step calls the predictor (which samples its prior), conditions
    autoregressively on the predicted state, and overwrites boundary nodes
    with the true value. This subclass adds :meth:`training_rollout`, the
    posterior-conditioned rollout used for training: at each step it asks the
    predictor for the prior, the variational posterior (conditioned on the
    true target) and the decoded reconstruction, autoregresses on the
    predicted mean, and reduces the per-step KL between posterior and prior.
    The KL is purely between the model's own distributions, so it is reduced
    here; the likelihood and the ELBO assembly are left to the module.
    """

    # The wrapped predictor must expose per-step latent distributions
    predictor: LatentStepPredictor

    def training_rollout(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        compute_kl: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Posterior-conditioned rollout producing the per-step ELBO components.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, d_state)``. ``[X_{t-1}, X_t]``.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_forcing)``.
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_state)``. The true next
            states, used to condition the posterior and to overwrite the
            boundary nodes.
        compute_kl : bool, optional
            If ``False`` (KL weight zero), skip the prior and return
            ``kl_terms`` as ``None``.

        Returns
        -------
        pred_means : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_state)``. Raw decoder
            means (not boundary-overwritten), for the likelihood.
        pred_stds : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, d_state)`` when the
            predictor emits its own std, otherwise ``None``.
        kl_terms : torch.Tensor or None
            Shape ``(B, pred_steps)``. Per-step KL of posterior from prior,
            or ``None`` when ``compute_kl`` is ``False``.
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        mean_list = []
        std_list = []
        kl_list = []

        for i in range(forcing_features.shape[1]):
            target = target_states[:, i]
            prior_dist, posterior_dist, pred_mean, pred_std = (
                self.predictor.step_distributions(
                    prev_state,
                    prev_prev_state,
                    forcing_features[:, i],
                    target=target,
                    compute_prior=compute_kl,
                )
            )

            mean_list.append(pred_mean)  # raw, masked to interior later
            if pred_std is not None:
                std_list.append(pred_std)
            if posterior_dist is not None and prior_dist is not None:
                kl = torch.distributions.kl_divergence(
                    posterior_dist, prior_dist
                )
                # Sum over all latent dimensions, leaving the batch dim
                kl_list.append(kl.sum(dim=tuple(range(1, kl.ndim))))

            # Autoregress on the predicted mean; boundary overwritten by truth
            new_state = (
                self.boundary_mask * target + self.interior_mask * pred_mean
            )
            prev_prev_state = prev_state
            prev_state = new_state

        pred_means = torch.stack(mean_list, dim=1)
        pred_stds = torch.stack(std_list, dim=1) if std_list else None
        kl_terms = torch.stack(kl_list, dim=1) if kl_list else None
        return pred_means, pred_stds, kl_terms

    def sample_trajectories(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
        num_traj: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample an ensemble of prior trajectories.

        Runs the evaluation rollout (:meth:`ARForecaster.forward`, which
        samples the prior each step) ``num_traj`` times and stacks the
        members along a new ensemble dimension. Used for the CRPS term and
        for ensemble evaluation.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, d_state)``. ``[X_{t-1}, X_t]``.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_forcing)``.
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_state)``. True states
            used to overwrite boundary nodes at each step.
        num_traj : int
            Number of trajectories ``S`` to sample.

        Returns
        -------
        traj_means : torch.Tensor
            Shape ``(B, S, pred_steps, num_grid_nodes, d_state)``.
        traj_stds : torch.Tensor or None
            Shape ``(B, S, pred_steps, num_grid_nodes, d_state)`` when the
            predictor emits its own std, otherwise ``None``.
        """
        mean_list = []
        std_list = []
        for _ in range(num_traj):
            prediction, pred_std = self(
                init_states, forcing_features, boundary_states
            )
            mean_list.append(prediction)
            if pred_std is not None:
                std_list.append(pred_std)

        traj_means = torch.stack(mean_list, dim=1)  # (B, S, T, N, d_state)
        traj_stds = torch.stack(std_list, dim=1) if std_list else None
        return traj_means, traj_stds
