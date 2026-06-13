"""Graph-based Ensemble Forecasting Model (Graph-EFM) single-step
predictors, for hierarchical (GraphEFM) and flat (GraphEFMMS) mesh
graphs."""

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


class BaseGraphEFM(StepPredictor):
    """
    Base class for Graph-based Ensemble Forecasting Model single-step
    predictors.

    A latent-variable step predictor consisting of a conditional prior, a
    variational encoder and a latent decoder, each of which carries its own
    grid-to-mesh, on-mesh and mesh-to-grid GNNs. The
    encode-process-decode backbone of
    ``BaseGraphModel`` therefore does not apply -- this extends
    ``StepPredictor`` directly. Besides ``forward`` (sampling a single step
    from the prior) it exposes the per-step ELBO pieces
    (``compute_step_loss`` -> ``(likelihood_term, kl_term, pred_mean,
    pred_std)``). Rollout, ELBO assembly, ensemble logic and logging live
    outside the predictor.

    This base class sets up everything that is independent of the mesh
    graph type. Concrete subclasses are specific to a graph type (declared
    by ``requires_hierarchical``): their constructors build the mesh
    embedders and the prior/encoder/decoder latent modules, and they
    implement :meth:`embedd_mesh`. See :class:`GraphEFM` (hierarchical
    graph) and :class:`GraphEFMMS` (flat graph).
    """

    # Whether the concrete subclass requires a hierarchical mesh graph
    requires_hierarchical: bool

    def __init__(
        self,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_name: str,
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        """
        Set up the graph-type independent parts of the predictor.

        Loads the graph, builds the grid embedders, the grid-mesh edge
        embedders and the constant per-variable std. Building the mesh
        embedders and the prior/encoder/decoder latent modules is left to
        the subclass constructor.

        Parameters
        ----------
        config : NeuralLAMConfig
            Full Neural-LAM configuration; used for the state feature
            weighting that enters the constant per-variable std.
        datastore : BaseDatastore
            Datastore providing static features, standardization statistics
            and variable counts.
        graph_name : str
            Name of the graph directory (under ``<root>/graph``) to load.
            Must be of the graph type required by the concrete subclass
            (``requires_hierarchical``).
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        num_past_forcing_steps : int
            Number of past forcing steps included in the input window.
        num_future_forcing_steps : int
            Number of future forcing steps included in the input window.
        output_std : bool
            If True, the decoder outputs a per-variable std alongside the
            mean; if False, a constant per-variable std is used as
            likelihood scale.
        output_clamping_lower : dict of str to float, optional
            Lower clamping limits per output variable.
        output_clamping_upper : dict of str to float, optional
            Upper clamping limits per output variable.
        """
        super().__init__(
            datastore=datastore,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        # Load graph with static features.
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices.
        self.hierarchical = utils.load_and_register_graph(
            self, datastore, graph_name
        )
        if self.hierarchical != self.requires_hierarchical:
            required_type = (
                "hierarchical" if self.requires_hierarchical else "flat"
            )
            loaded_type = "hierarchical" if self.hierarchical else "flat"
            raise ValueError(
                f"{type(self).__name__} requires a {required_type} mesh "
                f"graph, but graph '{graph_name}' is {loaded_type}"
            )

        # Specify dimensions of data
        self.num_state_vars = datastore.get_num_data_vars(category="state")
        num_state_vars = self.num_state_vars
        # grid_dim: total grid input dim. grid_current_dim additionally
        # includes the target state, for the encoder input.
        self.grid_dim = utils.grid_input_dim(
            datastore,
            self.grid_static_features.shape[1],
            num_past_forcing_steps,
            num_future_forcing_steps,
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

    def embedd_grid_with_target(
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

    def embedd_mesh(self, batch_size):
        """
        Embed static mesh node and intra-mesh edge features.

        Parameters
        ----------
        batch_size : int
            Batch size to expand the embeddings to.

        Returns
        -------
        dict
            Mesh-related entries of the graph embedding (``mesh``, ``m2m``
            and, for hierarchical graphs, ``mesh_up`` and ``mesh_down``).
            Entries are tensors of shape ``(B, *, d_h)`` for flat graphs
            and per-level lists of such tensors for hierarchical graphs.
        """
        raise NotImplementedError("embedd_mesh not implemented")

    def embedd_grid_and_graph(self, prev_state, prev_prev_state, forcing):
        """
        Embed the grid (states up to t-1) and the full graph.

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
        graph_emb.update(self.embedd_mesh(batch_size))

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
            ``embedd_grid_and_graph``.
        graph_emb : dict
            Edge/mesh embeddings from ``embedd_grid_and_graph``.
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
        grid_prev_emb, graph_emb = self.embedd_grid_and_graph(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
        )
        # embed also including current grid state, for encoder
        grid_current_emb = self.embedd_grid_with_target(
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
        Sample one time step prediction: embed features, sample the latent
        from the prior, decode, and return the predicted next state. The
        prediction is stochastic only through the latent sample; no
        observation noise is added.

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
            Shape ``(B, num_grid_nodes, d_state)``. Predicted ``X_{t+1}``
            (the decoder mean, given the sampled latent).
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, d_state)`` when ``output_std`` is True,
            otherwise None.
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_grid_and_graph(
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

        return pred_mean, pred_std


class GraphEFM(BaseGraphEFM):
    """
    Graph-based Ensemble Forecasting Model on a hierarchical mesh graph.

    The latent variable lives on the top level of the mesh hierarchy. The
    prior and variational encoder are ``HiGraphLatentEncoder``s and the
    decoder is a ``HiGraphLatentDecoder``.
    """

    requires_hierarchical = True

    def __init__(
        self,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_name: str = "hierarchical",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        latent_dim: Optional[int] = None,
        prior_intra_level_layers: int = 2,
        encoder_intra_level_layers: int = 2,
        decoder_intra_level_layers: int = 4,
        learn_prior: bool = True,
        prior_dist: str = "isotropic",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        """
        Build the mesh embedders and hierarchical latent modules.

        Parameters
        ----------
        config : NeuralLAMConfig
            Full Neural-LAM configuration; used for the state feature
            weighting that enters the constant per-variable std.
        datastore : BaseDatastore
            Datastore providing static features, standardization statistics
            and variable counts.
        graph_name : str
            Name of the graph directory (under ``<root>/graph``) to load.
            Must be a hierarchical graph.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        latent_dim : int, optional
            Dimensionality of the latent variable at each top-level mesh
            node; defaults to ``hidden_dim`` when None.
        prior_intra_level_layers : int
            Number of intra-level GNN layers in the (learned) prior.
        encoder_intra_level_layers : int
            Number of intra-level GNN layers in the variational encoder.
        decoder_intra_level_layers : int
            Number of intra-level GNN layers in the latent decoder.
        learn_prior : bool
            If True, the prior is a hierarchical graph encoder conditioned
            on the previous state; if False, a constant ``Normal(0, 1)``
            prior is used.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        num_past_forcing_steps : int
            Number of past forcing steps included in the input window.
        num_future_forcing_steps : int
            Number of future forcing steps included in the input window.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh steps of the prior, encoder and
            decoder (key in ``gnn_layers.GNN_TYPES``).
        m2g_gnn_type : str
            GNN type for the mesh-to-grid step of the decoder (key in
            ``gnn_layers.GNN_TYPES``).
        output_std : bool
            If True, the decoder outputs a per-variable std alongside the
            mean; if False, a constant per-variable std is used as
            likelihood scale.
        output_clamping_lower : dict of str to float, optional
            Lower clamping limits per output variable.
        output_clamping_upper : dict of str to float, optional
            Upper clamping limits per output variable.
        """
        super().__init__(
            config=config,
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]
        # The latent variable lives on the top mesh level
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
                utils.log_on_rank_zero(f"  {level_index}<->{level_index + 1}")
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
        # If not using any intra-level layers, no need to embed m2m
        self.embedd_m2m = (
            max(
                prior_intra_level_layers,
                encoder_intra_level_layers,
                decoder_intra_level_layers,
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

        latent_dim = latent_dim if latent_dim is not None else hidden_dim

        # Prior. When learn_prior, the prior is a graph encoder mapping the
        # previous state to a latent distribution; otherwise it is a constant
        # (input-independent) Normal.
        if learn_prior:
            self.prior_model = HiGraphLatentEncoder(
                latent_dim=latent_dim,
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                mesh_up_edge_index=self.mesh_up_edge_index,
                hidden_dim=hidden_dim,
                intra_level_layers=prior_intra_level_layers,
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

        # Encoder (variational posterior) + Decoder
        self.encoder = HiGraphLatentEncoder(
            latent_dim=latent_dim,
            g2m_edge_index=self.g2m_edge_index,
            m2m_edge_index=self.m2m_edge_index,
            mesh_up_edge_index=self.mesh_up_edge_index,
            hidden_dim=hidden_dim,
            intra_level_layers=encoder_intra_level_layers,
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
            num_state_vars=self.num_state_vars,
            intra_level_layers=decoder_intra_level_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
            output_std=bool(output_std),
        )

    def embedd_mesh(self, batch_size):
        """
        Embed static mesh node and intra-mesh edge features per level.

        Parameters
        ----------
        batch_size : int
            Batch size to expand the embeddings to.

        Returns
        -------
        dict
            Entries ``mesh``, ``m2m``, ``mesh_up`` and ``mesh_down``, each
            a list with one ``(B, *, d_h)`` tensor per mesh level (or
            inter-level connection).
        """
        mesh_emb = {
            "mesh": [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ],  # each (B, num_mesh_nodes[l], d_h)
            "mesh_up": [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ],
            "mesh_down": [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ],
        }

        if self.embedd_m2m:
            mesh_emb["m2m"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(self.m2m_embedders, self.m2m_features)
            ]
        else:
            # Need a placeholder otherwise, just use raw features
            mesh_emb["m2m"] = list(self.m2m_features)

        return mesh_emb


class GraphEFMMS(BaseGraphEFM):
    """
    Graph-based Ensemble Forecasting Model on a flat mesh graph
    (Graph-EFM-MS, e.g. for multi-scale graphs).

    The latent variable lives on the mesh nodes. The prior and variational
    encoder are ``GraphLatentEncoder``s and the decoder is a
    ``GraphLatentDecoder``.
    """

    requires_hierarchical = False

    def __init__(
        self,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_name: str = "multiscale",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        latent_dim: Optional[int] = None,
        prior_m2m_layers: int = 2,
        encoder_m2m_layers: int = 2,
        decoder_m2m_layers: int = 4,
        learn_prior: bool = True,
        prior_dist: str = "isotropic",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        """
        Build the mesh embedders and flat-graph latent modules.

        Parameters
        ----------
        config : NeuralLAMConfig
            Full Neural-LAM configuration; used for the state feature
            weighting that enters the constant per-variable std.
        datastore : BaseDatastore
            Datastore providing static features, standardization statistics
            and variable counts.
        graph_name : str
            Name of the graph directory (under ``<root>/graph``) to load.
            Must be a flat graph.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        latent_dim : int, optional
            Dimensionality of the latent variable at each mesh node;
            defaults to ``hidden_dim`` when None.
        prior_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the (learned) prior.
        encoder_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the variational encoder.
        decoder_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the latent decoder.
        learn_prior : bool
            If True, the prior is a graph encoder conditioned on the
            previous state; if False, a constant ``Normal(0, 1)`` prior is
            used.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        num_past_forcing_steps : int
            Number of past forcing steps included in the input window.
        num_future_forcing_steps : int
            Number of future forcing steps included in the input window.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh steps of the prior, encoder and
            decoder (key in ``gnn_layers.GNN_TYPES``).
        m2g_gnn_type : str
            GNN type for the mesh-to-grid step of the decoder (key in
            ``gnn_layers.GNN_TYPES``).
        output_std : bool
            If True, the decoder outputs a per-variable std alongside the
            mean; if False, a constant per-variable std is used as
            likelihood scale.
        output_clamping_lower : dict of str to float, optional
            Lower clamping limits per output variable.
        output_clamping_upper : dict of str to float, optional
            Upper clamping limits per output variable.
        """
        super().__init__(
            config=config,
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        self.num_mesh_nodes = self.mesh_static_features.shape[0]
        utils.log_on_rank_zero(
            f"Loaded graph with "
            f"{self.num_grid_nodes + self.num_mesh_nodes} nodes "
            f"({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # Embedders
        mesh_static_dim = self.mesh_static_features.shape[1]
        self.mesh_embedder = utils.make_mlp(
            [mesh_static_dim] + self.mlp_blueprint_end
        )
        m2m_dim = self.m2m_features.shape[1]
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        latent_dim = latent_dim if latent_dim is not None else hidden_dim

        # Prior. When learn_prior, the prior is a graph encoder mapping the
        # previous state to a latent distribution; otherwise it is a constant
        # (input-independent) Normal.
        if learn_prior:
            self.prior_model = GraphLatentEncoder(
                latent_dim=latent_dim,
                g2m_edge_index=self.g2m_edge_index,
                m2m_edge_index=self.m2m_edge_index,
                hidden_dim=hidden_dim,
                m2m_layers=prior_m2m_layers,
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

        # Encoder (variational posterior) + Decoder
        self.encoder = GraphLatentEncoder(
            latent_dim=latent_dim,
            g2m_edge_index=self.g2m_edge_index,
            m2m_edge_index=self.m2m_edge_index,
            hidden_dim=hidden_dim,
            m2m_layers=encoder_m2m_layers,
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
            num_state_vars=self.num_state_vars,
            m2m_layers=decoder_m2m_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
            output_std=bool(output_std),
        )

    def embedd_mesh(self, batch_size):
        """
        Embed static mesh node and intra-mesh edge features.

        Parameters
        ----------
        batch_size : int
            Batch size to expand the embeddings to.

        Returns
        -------
        dict
            Entries ``mesh``: ``(B, num_mesh_nodes, d_h)`` and
            ``m2m``: ``(B, M_m2m, d_h)``.
        """
        return {
            "mesh": self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            ),  # (B, num_mesh_nodes, d_h)
            "m2m": self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            ),  # (B, M_m2m, d_h)
        }
