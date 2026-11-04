"""Probabilistic forecaster training a Graph-EFM predictor via its ELBO."""

# Third-party
import torch

# Local
from ... import metrics
from ...config import NeuralLAMConfig
from ...datastore import BaseDatastore
from ...loss_weighting import get_per_var_std
from ..step_predictors.graph.graph_efm import BaseGraphEFM
from .probabilistic import BaseProbabilisticARForecaster


class GraphEFMForecaster(BaseProbabilisticARForecaster):
    """
    Auto-regressive ensemble forecaster for Graph-EFM step predictors.

    Wraps a :class:`BaseGraphEFM` predictor (hierarchical ``GraphEFM`` or
    flat ``GraphEFMMultiScale``), a latent-variable model consisting of a
    conditional prior, a variational encoder and a latent decoder. Forecast
    sampling (``forward``, ``sample_ensemble``) is inherited from
    :class:`BaseProbabilisticARForecaster`: each predictor call samples the
    prior and decodes, so unrolling produces one stochastic trajectory and
    stacking several gives an ensemble.

    This class supplies the training objective the base class leaves
    abstract: the evidence lower bound (ELBO). ``compute_training_loss``
    runs a variational rollout in which, at each step, the latent is drawn
    from the encoder (conditioned on the target), and accumulates a
    reconstruction likelihood term and a KL term between the encoder and the
    prior. The Gaussian likelihood, the constant per-variable std
    fallback ``self.per_var_std`` and the boundary/interior masks all live
    on the forecaster; the predictor only
    provides the network building blocks.
    """

    def __init__(
        self,
        predictor: BaseGraphEFM,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        kl_beta: float = 1.0,
        crps_weight: float = 0.0,
        crps_members: int = 2,
    ) -> None:
        """
        Initialize the GraphEFMForecaster.

        Parameters
        ----------
        predictor : BaseGraphEFM
            The Graph-EFM step predictor to use for each step. Each
            ``forward`` call samples the prior and decodes; the encoder,
            prior and decoder sub-models are also used directly by
            ``compute_training_loss`` to assemble the ELBO.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        config : NeuralLAMConfig or None
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when ``predictor`` does not output
            its own. Required for
            training when the predictor's ``output_std`` is False, since the
            likelihood term then needs the fallback std.
        crps_weight : float, optional
            Weight of fair CRPS computed from independent prior trajectories.
            Default zero, disabling the additional sampling cost.
        crps_members : int, optional
            Number of prior trajectories for CRPS. Default two.
        kl_beta : float, default 1.0
            Weight of the KL term in the ELBO. When ``0`` the prior and KL
            are not computed at all (pure auto-encoder training); the prior
            network then receives gradients only if CRPS is enabled.
        """
        super().__init__(predictor, datastore)
        if not torch.isfinite(torch.tensor(kl_beta)) or kl_beta < 0:
            raise ValueError("kl_beta must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(crps_weight)) or crps_weight < 0:
            raise ValueError("crps_weight must be finite and nonnegative")
        if crps_members < 2:
            raise ValueError("crps_members must be at least two")
        self.kl_beta = kl_beta
        self.crps_weight = crps_weight
        self.crps_members = crps_members
        self.register_buffer(
            "per_var_std",
            get_per_var_std(config, datastore) if config is not None else None,
            persistent=False,
        )

    predictor: BaseGraphEFM
    per_var_std: torch.Tensor | None

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the ELBO training objective for one batch.

        Unrolls a variational rollout over the full forecast: at each step
        the grid, graph and target are embedded, the latent is drawn from
        the encoder (variational posterior), the decoder reconstructs the
        next-state mean, and a reconstruction likelihood term is
        accumulated. When ``kl_beta`` is positive a KL term between the
        encoder and the prior is accumulated too. Both terms are summed over
        the rollout and averaged over the batch; the loss is
        ``-likelihood + kl_beta * kl + crps_weight * crps``.
        CRPS is
        summed over interior nodes, variables and time, like the likelihood,
        and averaged over the batch. The rollout advances on the
        predicted mean with boundary nodes overwritten by the true state.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the rollout from. Dims:
            ``B`` is batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes,
            num_windowed_forcing_vars)``.
            External forcings provided at each predicted step. Dims: ``B``
            is batch size, ``pred_steps`` is the rollout length,
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_windowed_forcing_vars`` includes the concatenated
            past/current/future forcing windows.
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            states at each predicted step, used as the encoder conditioning
            and reconstruction target and to overwrite boundary nodes during
            the rollout. Dims: same as the prediction.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``metrics.nll`` so only interior
            nodes contribute to the likelihood.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. Negative weighted ELBO plus optional fair CRPS.
        loss_components : dict of {str: torch.Tensor}
            Scalar ELBO diagnostics to log: ``"elbo_likelihood"`` always,
            plus ``"elbo_kl"`` and ``"elbo"`` when ``kl_beta > 0``, and
            ``"crps"`` when ``crps_weight > 0``.

        Raises
        ------
        ValueError
            If the predictor does not output its own std and no
            ``per_var_std`` fallback is available (this forecaster was
            constructed without ``config``).
        """
        predictor = self.predictor

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        pred_steps = forcing_features.shape[1]
        compute_kl = self.kl_beta > 0

        # The graph embedding depends only on static features, so it is
        # constant across the rollout and computed once here.
        graph_emb = predictor.embedd_graph(init_states.shape[0])

        likelihood_terms = []
        kl_terms = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            target_state = target_states[:, i]

            # Posterior latent (conditioned on target), reconstruction, and
            # -- when a KL term is needed -- the prior, in one predictor call.
            prior_dist, posterior_dist, pred_mean, pred_std = (
                predictor.posterior_step(
                    prev_state,
                    prev_prev_state,
                    forcing,
                    target_state,
                    graph_emb,
                    compute_prior=compute_kl,
                )
            )
            if pred_std is None:
                pred_std = self.per_var_std
            if pred_std is None:
                raise ValueError(
                    "Graph-EFM likelihood requires config or output_std"
                )

            # Reconstruction likelihood, summed over interior grid and vars
            entry_likelihoods = -metrics.nll(
                pred_mean,
                target_state,
                pred_std,
                mask=interior_mask_bool,
                average_grid=False,
                sum_vars=False,
            )  # (B, num_interior_grid_nodes, num_state_vars)
            likelihood_terms.append(torch.sum(entry_likelihoods, dim=(1, 2)))

            if compute_kl:
                assert prior_dist is not None
                kl_terms.append(
                    torch.sum(
                        torch.distributions.kl_divergence(
                            posterior_dist, prior_dist
                        ),
                        dim=(1, 2),
                    )
                )  # (B,)

            # Advance the rollout on the predicted mean, boundary overwritten
            new_state = (
                self.boundary_mask * target_state
                + self.interior_mask * pred_mean
            )
            prev_prev_state = prev_state
            prev_state = new_state

        # Sum each term over the rollout, then average over the batch
        mean_likelihood = torch.mean(
            torch.sum(torch.stack(likelihood_terms, dim=1), dim=1)
        )
        loss_components = {"elbo_likelihood": mean_likelihood}

        if compute_kl:
            mean_kl = torch.mean(torch.sum(torch.stack(kl_terms, dim=1), dim=1))
            batch_loss = -mean_likelihood + self.kl_beta * mean_kl
            loss_components["elbo_kl"] = mean_kl
            loss_components["elbo"] = mean_likelihood - mean_kl
        else:
            batch_loss = -mean_likelihood

        if self.crps_weight > 0:
            ensemble, _ = self.sample_ensemble(
                init_states, forcing_features, target_states, self.crps_members
            )
            entry_crps = metrics.crps_ens_fair(
                ensemble,
                target_states,
                mask=interior_mask_bool,
                average_grid=False,
                sum_vars=False,
            )
            crps = entry_crps.sum(dim=(1, 2, 3)).mean()
            batch_loss = batch_loss + self.crps_weight * crps
            loss_components["crps"] = crps
        return batch_loss, loss_components
