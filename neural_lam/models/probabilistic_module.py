"""Lightning module that trains latent-variable forecasters with the ELBO."""

# Local
from .forecasters.probabilistic import ProbabilisticARForecaster
from .module import ForecasterModule


class ProbabilisticForecasterModule(ForecasterModule):
    """
    Lightning module for latent-variable forecasters.

    Extends :class:`ForecasterModule` by assembling the training loss as the
    (beta-weighted) negative ELBO. The forecaster's ``training_rollout``
    returns the per-step decoder reconstruction and the per-step KL; this
    module computes the likelihood from the reconstruction and the target
    through ``self.loss`` (substituting the constant ``per_var_std`` when the
    predictor emits no std), sums the likelihood and KL over the rollout,
    weights the KL by ``kl_beta`` and averages over the batch. Evaluation
    reuses the inherited single-member rollout.
    """

    # The wrapped forecaster must roll out a latent-variable predictor
    forecaster: ProbabilisticARForecaster

    def __init__(self, *args, kl_beta: float = 1.0, **kwargs):
        """
        Initialize the module and store the KL weight.

        Parameters
        ----------
        *args
            Positional arguments forwarded to
            :meth:`ForecasterModule.__init__` (``forecaster``, ``config``,
            ``datastore``, ...).
        kl_beta : float, optional
            Weight of the KL term in the ELBO. ``0`` trains a pure
            autoencoder (the prior and KL are skipped). Default ``1.0``.
        **kwargs
            Keyword arguments forwarded to
            :meth:`ForecasterModule.__init__` (``loss``, ``lr``, ...).
        """
        super().__init__(*args, **kwargs)
        self.kl_beta = kl_beta

    def training_step(self, batch):
        """
        Assemble the beta-weighted negative ELBO over the rollout.

        Parameters
        ----------
        batch : tuple
            ``(init_states, target_states, forcing_features, batch_times)``.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        init_states, target_states, forcing_features, _ = batch
        compute_kl = self.kl_beta > 0

        pred_means, pred_stds, kl_terms = self.forecaster.training_rollout(
            init_states,
            forcing_features,
            target_states,
            compute_kl=compute_kl,
        )
        pred_std = pred_stds if pred_stds is not None else self.per_var_std

        # Per-entry log-likelihood is the negative loss (exactly the
        # log-likelihood for the nll loss); sum over interior grid + vars.
        entry_log_lik = -self.loss(
            pred_means,
            target_states,
            pred_std,
            mask=self.interior_mask_bool,
            average_grid=False,
            sum_vars=False,
        )  # (B, pred_steps, num_interior_nodes, d_state)
        likelihood_terms = entry_log_lik.sum(dim=(2, 3))  # (B, pred_steps)

        # Sum the per-step terms over the rollout, mean over the batch
        per_sample_likelihood = likelihood_terms.sum(dim=1)  # (B,)
        loss = -per_sample_likelihood.mean()
        log_dict = {"elbo_likelihood": per_sample_likelihood.mean()}
        if kl_terms is not None:
            per_sample_kl = kl_terms.sum(dim=1)  # (B,)
            loss = loss + self.kl_beta * per_sample_kl.mean()
            log_dict["elbo_kl"] = per_sample_kl.mean()
        log_dict["train_loss"] = loss

        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=init_states.shape[0],
        )
        return loss
