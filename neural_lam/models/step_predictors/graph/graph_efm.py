# Standard library
from typing import Callable, Dict, Optional

# Third-party
import torch
from torch import nn

# Local
from .... import utils
from ....config import NeuralLAMConfig
from ....datastore import BaseDatastore
from ....loss_weighting import get_state_feature_weighting
from ...latent import (
    ConstantLatentEncoder,
    GraphLatentDecoder,
    GraphLatentEncoder,
    HiGraphLatentDecoder,
    HiGraphLatentEncoder,
)
from ..base import StepPredictor


class GraphEFM(StepPredictor):
    """
    Graph-based Ensemble Forecasting Model -- single-step predictor.

    Port of ``prob_model_lam``'s ``GraphEFM`` (``forward`` is the source's
    ``predict_step``) onto the ``StepPredictor`` interface. The predictor owns
    its own conditional-prior / variational-encoder / latent-decoder, each of
    which carries its own g2m/processor/m2g GNNs, so the encode-process-decode
    backbone of ``BaseGraphModel`` does not apply -- this extends
    ``StepPredictor`` directly. It is self-contained: besides ``forward`` it
    exposes the per-step ELBO pieces (``compute_step_loss`` ->
    ``(likelihood_term, kl_term, pred_mean, pred_std)``) and the sampling
    helpers used by a future rollout/ensemble module. Rollout, ELBO assembly,
    ensemble logic and logging live outside the predictor.

    One class handles both flat and hierarchical meshes, resolved at
    construction from ``self.hierarchical`` (set by ``utils.load_graph``).
    """

    def __init__(
        self,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_name: str = "hierarchical",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        latent_dim: Optional[int] = None,
        prior_processor_layers: int = 2,
        encoder_processor_layers: int = 2,
        processor_layers: int = 4,
        learn_prior: bool = True,
        prior_dist: str = "isotropic",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
        output_std: bool = False,
        sample_obs_noise: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            datastore=datastore,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        # Whether to sample observation noise during rollout. When False,
        # sample_next_state returns the predicted mean.
        self.sample_obs_noise = bool(sample_obs_noise)

        # Load graph with static features (same pattern as BaseGraphModel).
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices.
        graph_dir_path = datastore.root_path / "graph" / graph_name
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data (datastore-driven; replaces source's
        # constants.GRID_STATE_DIM / GRID_FORCING_DIM).
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        grid_static_dim = self.grid_static_features.shape[1]
        # grid_dim: total grid input dim, same formula as BaseGraphModel. The
        # cat ORDER in embedd_all/embedd_current follows source
        # (prev_prev, prev, forcing, static[, current]); the size is unchanged.
        self.grid_dim = (
            2 * num_state_vars
            + grid_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )
        grid_current_dim = self.grid_dim + num_state_vars
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.grid_prev_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )  # For states up to t-1
        self.grid_current_embedder = utils.make_mlp(
            [grid_current_dim] + self.mlp_blueprint_end
        )  # For states including t
        # Embedders for mesh edges
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        if self.hierarchical:
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            self.num_mesh_nodes = level_mesh_sizes[-1]
            num_levels = len(self.mesh_static_features)
            utils.log_on_rank_zero("Loaded hierarchical graph with structure:")
            for level_index, level_mesh_size in enumerate(level_mesh_sizes):
                same_level_edges = self.m2m_features[level_index].shape[0]
                utils.log_on_rank_zero(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )
                if level_index < (num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    utils.log_on_rank_zero(
                        f"  {level_index}<->{level_index + 1}"
                    )
                    utils.log_on_rank_zero(
                        f" - {up_edges} up edges, {down_edges} down edges"
                    )

            # Embedders. Assume all levels share static feature dimensionality.
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            self.mesh_embedders = nn.ModuleList(
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
                ]
            )
            self.mesh_up_embedders = nn.ModuleList(
                [
                    utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_down_embedders = nn.ModuleList(
                [
                    utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            # If not using any processor layers, no need to embed m2m
            self.embedd_m2m = (
                max(
                    prior_processor_layers,
                    encoder_processor_layers,
                    processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = nn.ModuleList(
                    [
                        utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            self.num_mesh_nodes = self.mesh_static_features.shape[0]
            utils.log_on_rank_zero(
                f"Loaded graph with "
                f"{self.num_grid_nodes + self.num_mesh_nodes} nodes "
                f"({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
            )
            mesh_static_dim = self.mesh_static_features.shape[1]
            self.mesh_embedder = utils.make_mlp(
                [mesh_static_dim] + self.mlp_blueprint_end
            )
            m2m_dim = self.m2m_features.shape[1]
            self.m2m_embedder = utils.make_mlp(
                [m2m_dim] + self.mlp_blueprint_end
            )

        latent_dim = latent_dim if latent_dim is not None else hidden_dim

        # Prior. When learn_prior, the prior is a graph encoder mapping the
        # previous state to a latent distribution; otherwise it is a constant
        # (input-independent) Normal.
        if learn_prior:
            if self.hierarchical:
                self.prior_model = HiGraphLatentEncoder(
                    latent_dim=latent_dim,
                    g2m_edge_index=self.g2m_edge_index,
                    m2m_edge_index=self.m2m_edge_index,
                    mesh_up_edge_index=self.mesh_up_edge_index,
                    hidden_dim=hidden_dim,
                    intra_level_layers=prior_processor_layers,
                    hidden_layers=hidden_layers,
                    g2m_gnn_type=g2m_gnn_type,
                    output_dist=prior_dist,
                )
            else:
                self.prior_model = GraphLatentEncoder(
                    latent_dim=latent_dim,
                    g2m_edge_index=self.g2m_edge_index,
                    m2m_edge_index=self.m2m_edge_index,
                    hidden_dim=hidden_dim,
                    m2m_layers=prior_processor_layers,
                    hidden_layers=hidden_layers,
                    g2m_gnn_type=g2m_gnn_type,
                    output_dist=prior_dist,
                )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim=latent_dim,
                num_mesh_nodes=self.num_mesh_nodes,
                output_dist=prior_dist,
            )

        # Encoder (variational posterior) + Decoder. The latent modules take
        # num_state_vars (datastore-driven) where source used GRID_STATE_DIM.
        if self.hierarchical:
            self.encoder = HiGraphLatentEncoder(
                latent_dim=latent_dim,
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                mesh_up_edge_index=self.mesh_up_edge_index,
                hidden_dim=hidden_dim,
                intra_level_layers=encoder_processor_layers,
                hidden_layers=hidden_layers,
                g2m_gnn_type=g2m_gnn_type,
                output_dist="diagonal",
            )
            self.decoder = HiGraphLatentDecoder(
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                m2g_edge_index=self.m2g_edge_index,
                mesh_up_edge_index=self.mesh_up_edge_index,
                mesh_down_edge_index=self.mesh_down_edge_index,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_state_vars=num_state_vars,
                intra_level_layers=processor_layers,
                hidden_layers=hidden_layers,
                g2m_gnn_type=g2m_gnn_type,
                m2g_gnn_type=m2g_gnn_type,
                output_std=bool(output_std),
            )
        else:
            self.encoder = GraphLatentEncoder(
                latent_dim=latent_dim,
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                hidden_dim=hidden_dim,
                m2m_layers=encoder_processor_layers,
                hidden_layers=hidden_layers,
                g2m_gnn_type=g2m_gnn_type,
                output_dist="diagonal",
            )
            self.decoder = GraphLatentDecoder(
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                m2g_edge_index=self.m2g_edge_index,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_state_vars=num_state_vars,
                m2m_layers=processor_layers,
                hidden_layers=hidden_layers,
                g2m_gnn_type=g2m_gnn_type,
                m2g_gnn_type=m2g_gnn_type,
                output_std=bool(output_std),
            )

        # Constant per-variable std used as the (homoscedastic) likelihood
        # scale when the decoder does not output its own std. Mirrors
        # ForecasterModule's per_var_std formula
        # (state_diff_std_standardized / sqrt(state_feature_weights)); both
        # copies are persistent=False so there is no checkpoint interaction.
        if not self.output_std:
            da_state_stats = datastore.get_standardization_dataarray(
                category="state"
            )
            state_diff_std = torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            )
            state_feature_weights = torch.tensor(
                get_state_feature_weighting(config=config, datastore=datastore),
                dtype=torch.float32,
            )
            self.register_buffer(
                "per_var_std",
                state_diff_std / torch.sqrt(state_feature_weights),
                persistent=False,
            )
        else:
            self.per_var_std = None

        # Compute indices and define clamping functions. GraphEFM's forward
        # never clamps (the decoder outputs the full next state), so these are
        # inert -- accepted for interface parity with other StepPredictors.
        self.prepare_clamping_params(datastore)

    def sample_next_state(self, pred_mean, pred_std):
        """
        Sample state at next time step given a Gaussian observation model.
        If ``self.sample_obs_noise`` is False, only return the mean.

        Parameters
        ----------
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. Predicted mean.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, d_state)``, or None when the decoder
            does not output a std (``output_std=False``).

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. Next state.
        """
        if not self.output_std:
            pred_std = self.per_var_std  # (d_f,)

        if self.sample_obs_noise:
            return torch.distributions.Normal(pred_mean, pred_std).rsample()
            # (B, num_grid_nodes, d_state)

        return pred_mean  # (B, num_grid_nodes, d_state)

    def embedd_current(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        current_state,
    ):
        """
        Embed the grid representation including the current (target) state.
        Used as input to the encoder, which is conditioned also on the target.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, d_forcing)``.
        current_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_{t+1}`` (target).

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid embedding.
        """
        batch_size = prev_state.shape[0]

        grid_current_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
                current_state,
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_current_dim)

        return self.grid_current_embedder(
            grid_current_features
        )  # (B, num_grid_nodes, d_h)

    def embedd_all(self, prev_state, prev_prev_state, forcing):
        """
        Embed all node and edge representations.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, d_forcing)``.

        Returns
        -------
        grid_emb : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid embedding.
        graph_emb : dict
            Edge/mesh embeddings, each entry of shape ``(B, *, d_h)``.
        """
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder(grid_features)
        # (B, num_grid_nodes, d_h)

        # Graph embedding. NOTE: this block depends only on static graph
        # features, so it is constant across an autoregressive rollout. It is
        # kept as a self-contained block so a future embedd_graph()/
        # embedd_grid() split (hoisting it out of the AR loop) is mechanical.
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(
                        self.m2m_embedders, self.m2m_features
                    )
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return grid_emb, graph_emb

    def estimate_likelihood(
        self,
        latent_dist,
        current_state,
        last_state,
        grid_prev_emb,
        graph_emb,
        loss_fn: Callable,
        interior_mask: torch.Tensor,
    ):
        """
        Estimate the (masked) likelihood using the given distribution over
        latent variables.

        ``loss_fn`` and ``interior_mask`` are passed in (not stored on the
        predictor): masks live on the forecaster/module, which supplies its
        own loss function and boolean interior mask.

        Parameters
        ----------
        latent_dist : torch.distributions.Distribution
            Shape ``(B, num_mesh_nodes, d_latent)``.
        current_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. Target ``X_{t+1}``.
        last_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_t``.
        grid_prev_emb : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid embedding from
            ``embedd_all``.
        graph_emb : dict
            Edge/mesh embeddings from ``embedd_all``.
        loss_fn : Callable
            Per-entry loss (e.g. ``metrics.nll``); likelihood is its negative.
        interior_mask : torch.Tensor
            Boolean ``(num_grid_nodes,)`` mask of interior nodes.

        Returns
        -------
        likelihood_term : torch.Tensor
            Shape ``(B,)``.
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``.
        pred_std : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)`` (decoder) or ``(d_state,)``
            (constant ``per_var_std``).
        """
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, model_pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # both (B, num_grid_nodes, d_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute likelihood (negative loss, exactly likelihood for nll loss)
        # Note: There are some round-off errors here due to float32
        # and large values
        entry_likelihoods = -loss_fn(
            pred_mean,
            current_state,
            pred_std,
            mask=interior_mask,
            average_grid=False,
            sum_vars=False,
        )  # (B, num_grid_nodes', d_state)
        likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
        return likelihood_term, pred_mean, pred_std

    def compute_step_loss(
        self,
        prev_states,
        current_state,
        forcing_features,
        loss_fn: Callable,
        interior_mask: torch.Tensor,
        compute_kl: bool = True,
    ):
        """
        Forward pass and per-step ELBO pieces for one time step.

        Parameters
        ----------
        prev_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, d_state)``. ``X_{t-1}, X_t``.
        current_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. Target ``X_{t+1}``.
        forcing_features : torch.Tensor
            Shape ``(B, num_grid_nodes, d_forcing)``.
        loss_fn : Callable
            Per-entry loss used to compute the likelihood term.
        interior_mask : torch.Tensor
            Boolean ``(num_grid_nodes,)`` mask of interior nodes.
        compute_kl : bool
            When False, skip the prior and return ``kl_term = None`` (the
            ``kl_beta == 0`` / pure-autoencoder case). The KL weight itself is
            a training knob owned by the calling module.

        Returns
        -------
        likelihood_term : torch.Tensor
            Shape ``(B,)``.
        kl_term : torch.Tensor or None
            Shape ``(B,)``, or None when ``compute_kl`` is False.
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``.
        pred_std : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)`` or ``(d_state,)``.
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
        )
        # embed also including current grid state, for encoder
        grid_current_emb = self.embedd_current(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
            current_state,
        )  # (B, num_grid_nodes, d_h)

        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            grid_current_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        last_state = prev_states[:, -1]
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist,
            current_state,
            last_state,
            grid_prev_emb,
            graph_emb,
            loss_fn,
            interior_mask,
        )
        if compute_kl:
            # Compute prior
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Compute KL
            kl_term = torch.sum(
                torch.distributions.kl_divergence(var_dist, prior_dist),
                dim=(1, 2),
            )  # (B,)
        else:
            # If KL is off, do not need to even compute prior nor KL
            kl_term = None  # Set to None to crash if erroneously used

        return likelihood_term, kl_term, pred_mean, pred_std

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample one time step prediction (source's ``predict_step``):
        embed features, sample the latent from the prior, decode, and return
        the sampled next state.

        Parameters
        ----------
        prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_t``.
        prev_prev_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. ``X_{t-1}``.
        forcing : torch.Tensor
            Shape ``(B, num_grid_nodes, d_forcing)``.

        Returns
        -------
        new_state : torch.Tensor
            Shape ``(B, num_grid_nodes, d_state)``. Sampled ``X_{t+1}``.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, d_state)`` when ``output_std`` is True,
            otherwise None.
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(
            prev_state, prev_prev_state, forcing
        )

        # Compute prior
        prior_dist = self.prior_model(
            grid_prev_emb, graph_emb=graph_emb
        )  # (B, num_mesh_nodes, d_latent)

        # Sample from prior
        latent_samples = prior_dist.rsample()
        # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        last_state = prev_state
        pred_mean, pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # (B, num_grid_nodes, d_state)

        return self.sample_next_state(pred_mean, pred_std), pred_std
