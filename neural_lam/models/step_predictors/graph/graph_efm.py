"""Graph-based Ensemble Forecasting Model (Graph-EFM) single-step
predictors, for hierarchical (GraphEFM) and flat (GraphEFMMultiScale) mesh
graphs."""

# Standard library
from typing import Dict, Optional

# Third-party
import torch
from torch import nn

# Local
from .... import utils
from ....datastore import BaseDatastore
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
    ``StepPredictor`` directly. ``forward`` samples a single step from the
    prior. The encoder (variational posterior) and prior are exposed as
    ``self.encoder``/``self.prior_model`` for a forecaster to condition on
    the target and assemble a training objective from; the predictor itself
    does not compute any loss.

    This base class sets up everything that is independent of the mesh
    graph type: it loads the graph, calls the subclass's
    :meth:`check_graph_type` to verify the loaded graph matches what it
    requires, then builds the prior, delegating to the subclass's
    :meth:`build_learnable_prior` when learned and to
    :attr:`latent_spatial_dim` for the constant prior's node count.
    Concrete subclasses build the mesh embedders and the encoder/decoder
    latent modules, and implement :meth:`embedd_mesh`,
    :meth:`check_graph_type`, :meth:`build_learnable_prior` and
    :attr:`latent_spatial_dim`. See :class:`GraphEFM` (hierarchical graph)
    and :class:`GraphEFMMultiScale` (flat graph).
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        graph_name: str,
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        latent_dim: Optional[int] = None,
        learn_prior: bool = True,
        prior_dist: str = "isotropic",
        prior_layers: int = 2,
        g2m_gnn_type: str = "InteractionNet",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        """
        Set up the graph-type independent parts of the predictor.

        Loads the graph, builds the grid embedders, the grid-mesh edge
        embedders and the prior. Building the mesh embedders and the
        encoder/decoder latent modules is left to the subclass constructor.

        Parameters
        ----------
        datastore : BaseDatastore
            Datastore providing static features, standardization statistics
            and variable counts.
        graph_name : str
            Name of the graph directory (under ``<root>/graph``) to load.
            Must be of the graph type required by the concrete subclass.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        latent_dim : int, optional
            Dimensionality of the latent variable at each latent-carrying
            mesh node (top level for hierarchical graphs, all mesh nodes
            for flat graphs); defaults to ``hidden_dim`` when None. The
            resolved value is stored as ``self.latent_dim`` for the
            subclass to reuse when building its encoder/decoder.
        learn_prior : bool
            If True, the prior is the graph-type specific learnable encoder
            built by :meth:`build_learnable_prior`, conditioned on the
            previous state; if False, a constant ``Normal(0, 1)`` prior is
            used.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        prior_layers : int
            Number of on-mesh GNN layers in the learnable prior.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step of the prior (key in
            ``gnn_layers.GNN_TYPES``).
        num_past_forcing_steps : int
            Number of past forcing steps included in the input window.
        num_future_forcing_steps : int
            Number of future forcing steps included in the input window.
        output_std : bool
            If True, the decoder outputs a per-variable std alongside the
            mean; if False, ``forward`` returns ``None`` for the std.
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
        grid_xy_extent = datastore.get_xy_extent(category="state")
        grid_xy_max_span = max(
            grid_xy_extent[1] - grid_xy_extent[0],
            grid_xy_extent[3] - grid_xy_extent[2],
        )
        self.hierarchical = utils.load_and_register_graph(
            self,
            datastore,
            graph_name,
            mesh_node_features_scaling=grid_xy_max_span,
        )
        # Delegated to the subclass, which knows whether it requires a
        # hierarchical or flat mesh graph. Must run before anything below
        # that assumes a specific graph shape (build_learnable_prior,
        # latent_spatial_dim), so it cannot wait until the subclass
        # constructor resumes after this call returns.
        self.check_graph_type(graph_name)

        # Specify dimensions of data
        self.num_state_vars = datastore.get_num_data_vars(category="state")
        num_state_vars = self.num_state_vars
        # grid_dim: total grid input dim. grid_current_dim additionally
        # includes the target state, for the encoder input.
        self.grid_dim = utils.compute_grid_input_dim(
            datastore,
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

        # Compute indices and define clamping functions. GraphEFM's forward
        # never clamps (the decoder outputs the full next state), so these are
        # inert -- accepted for interface parity with other StepPredictors.
        self.prepare_clamping_params(datastore)

        # Prior over the latent variable. When learn_prior is True the
        # (graph-type specific) learnable prior is delegated to
        # build_learnable_prior; otherwise the constant Normal(0, 1) prior,
        # identical for every graph type, is built directly here.
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim
        if learn_prior:
            self.prior_model = self.build_learnable_prior(
                latent_dim=self.latent_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                g2m_gnn_type=g2m_gnn_type,
                prior_dist=prior_dist,
                prior_layers=prior_layers,
            )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim=self.latent_dim,
                num_mesh_nodes=self.latent_spatial_dim,
                output_dist=prior_dist,
            )

    def check_graph_type(self, graph_name: str) -> None:
        """
        Verify the loaded graph (``self.hierarchical``) is of the type this
        predictor requires.

        Implemented by the concrete subclass, which is the only place that
        knows whether it requires a hierarchical or flat mesh graph. Called
        by this base class right after loading the graph, before anything
        that assumes a specific graph shape.

        Parameters
        ----------
        graph_name : str
            Name of the graph directory that was loaded, for the error
            message.

        Raises
        ------
        ValueError
            If ``self.hierarchical`` does not match what this predictor
            requires.
        """
        raise NotImplementedError("check_graph_type not implemented")

    @property
    def latent_spatial_dim(self) -> int:
        """
        Number of mesh nodes the latent variable lives on.

        Implemented by the concrete subclass, which knows the mesh graph
        type: the top mesh level for hierarchical graphs, or every mesh
        node for flat graphs.

        Returns
        -------
        int
            Number of latent-carrying mesh nodes.
        """
        raise NotImplementedError("latent_spatial_dim not implemented")

    def build_learnable_prior(
        self,
        latent_dim,
        hidden_dim,
        hidden_layers,
        g2m_gnn_type,
        prior_dist,
        prior_layers,
    ):
        """
        Build the graph-type specific learnable prior encoder.

        Implemented by the concrete subclass, which knows the mesh graph type
        and therefore the appropriate latent encoder class.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step of the prior.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        prior_layers : int
            Number of on-mesh GNN layers in the prior.

        Returns
        -------
        torch.nn.Module
            The learnable prior latent encoder.
        """
        raise NotImplementedError("build_learnable_prior not implemented")

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

        # Compute reconstruction (decoder). prev_state (X_t) is the state the
        # decoder adds its predicted residual onto.
        pred_mean, pred_std = self.decoder(
            grid_prev_emb, latent_samples, prev_state, graph_emb
        )  # (B, num_grid_nodes, d_state)

        return pred_mean, pred_std


class GraphEFM(BaseGraphEFM):
    """
    Graph-based Ensemble Forecasting Model on a hierarchical mesh graph.

    The latent variable lives on the top level of the mesh hierarchy. The
    prior and variational encoder are ``HiGraphLatentEncoder``s and the
    decoder is a ``HiGraphLatentDecoder``.
    """

    def __init__(
        self,
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
        Build the mesh embedders and the hierarchical encoder/decoder
        latent modules. The prior is built by the base class.

        Parameters
        ----------
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
            node; defaults to ``hidden_dim`` when None. Forwarded to the
            base class, which resolves the default and stores it as
            ``self.latent_dim``.
        prior_intra_level_layers : int
            Number of intra-level GNN layers in the (learned) prior.
            Forwarded to the base class as ``prior_layers``.
        encoder_intra_level_layers : int
            Number of intra-level GNN layers in the variational encoder.
        decoder_intra_level_layers : int
            Number of intra-level GNN layers in the latent decoder.
        learn_prior : bool
            If True, the prior is a hierarchical graph encoder conditioned
            on the previous state; if False, a constant ``Normal(0, 1)``
            prior is used. Forwarded to the base class.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``. Forwarded to the base class.
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
            mean; if False, ``forward`` returns ``None`` for the std.
        output_clamping_lower : dict of str to float, optional
            Lower clamping limits per output variable.
        output_clamping_upper : dict of str to float, optional
            Upper clamping limits per output variable.
        """
        super().__init__(
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            latent_dim=latent_dim,
            learn_prior=learn_prior,
            prior_dist=prior_dist,
            prior_layers=prior_intra_level_layers,
            g2m_gnn_type=g2m_gnn_type,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]
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

        # Encoder (variational posterior) + Decoder
        self.encoder = HiGraphLatentEncoder(
            latent_dim=self.latent_dim,
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
            latent_dim=self.latent_dim,
            num_state_vars=self.num_state_vars,
            intra_level_layers=decoder_intra_level_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
            output_std=bool(output_std),
        )

    def check_graph_type(self, graph_name: str) -> None:
        """
        Verify the loaded graph is hierarchical.

        Parameters
        ----------
        graph_name : str
            Name of the graph directory that was loaded, for the error
            message.

        Raises
        ------
        ValueError
            If the loaded graph is flat.
        """
        if not self.hierarchical:
            raise ValueError(
                f"{type(self).__name__} requires a hierarchical mesh "
                f"graph, but graph '{graph_name}' is flat"
            )

    @property
    def latent_spatial_dim(self) -> int:
        """
        Number of mesh nodes on the top mesh level, where the latent
        variable lives.

        Returns
        -------
        int
            Number of top-level mesh nodes.
        """
        return self.mesh_static_features[-1].shape[0]

    def build_learnable_prior(
        self,
        latent_dim,
        hidden_dim,
        hidden_layers,
        g2m_gnn_type,
        prior_dist,
        prior_layers,
    ):
        """
        Build the hierarchical learnable prior encoder.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each top-level mesh node.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step of the prior.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        prior_layers : int
            Number of intra-level GNN layers in the prior.

        Returns
        -------
        HiGraphLatentEncoder
            The learnable prior latent encoder.
        """
        return HiGraphLatentEncoder(
            latent_dim=latent_dim,
            g2m_edge_index=self.g2m_edge_index,
            m2m_edge_index=self.m2m_edge_index,
            mesh_up_edge_index=self.mesh_up_edge_index,
            hidden_dim=hidden_dim,
            intra_level_layers=prior_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            output_dist=prior_dist,
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


class GraphEFMMultiScale(BaseGraphEFM):
    """
    Graph-based Ensemble Forecasting Model on a flat mesh graph
    (e.g. a multi-scale graph).

    The latent variable lives on the mesh nodes. The prior and variational
    encoder are ``GraphLatentEncoder``s and the decoder is a
    ``GraphLatentDecoder``.
    """

    def __init__(
        self,
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
        Build the mesh embedders and the flat-graph encoder/decoder latent
        modules. The prior is built by the base class.

        Parameters
        ----------
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
            defaults to ``hidden_dim`` when None. Forwarded to the base
            class, which resolves the default and stores it as
            ``self.latent_dim``.
        prior_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the (learned) prior.
            Forwarded to the base class as ``prior_layers``.
        encoder_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the variational encoder.
        decoder_m2m_layers : int
            Number of on-mesh (m2m) GNN layers in the latent decoder.
        learn_prior : bool
            If True, the prior is a graph encoder conditioned on the
            previous state; if False, a constant ``Normal(0, 1)`` prior is
            used. Forwarded to the base class.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``. Forwarded to the base class.
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
            mean; if False, ``forward`` returns ``None`` for the std.
        output_clamping_lower : dict of str to float, optional
            Lower clamping limits per output variable.
        output_clamping_upper : dict of str to float, optional
            Upper clamping limits per output variable.
        """
        super().__init__(
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            latent_dim=latent_dim,
            learn_prior=learn_prior,
            prior_dist=prior_dist,
            prior_layers=prior_m2m_layers,
            g2m_gnn_type=g2m_gnn_type,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        utils.log_on_rank_zero(
            f"Loaded graph with "
            f"{self.num_grid_nodes + self.latent_spatial_dim} nodes "
            f"({self.num_grid_nodes} grid, {self.latent_spatial_dim} mesh)"
        )

        # Embedders
        mesh_static_dim = self.mesh_static_features.shape[1]
        self.mesh_embedder = utils.make_mlp(
            [mesh_static_dim] + self.mlp_blueprint_end
        )
        m2m_dim = self.m2m_features.shape[1]
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # Encoder (variational posterior) + Decoder
        self.encoder = GraphLatentEncoder(
            latent_dim=self.latent_dim,
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
            latent_dim=self.latent_dim,
            num_state_vars=self.num_state_vars,
            m2m_layers=decoder_m2m_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
            output_std=bool(output_std),
        )

    def check_graph_type(self, graph_name: str) -> None:
        """
        Verify the loaded graph is flat.

        Parameters
        ----------
        graph_name : str
            Name of the graph directory that was loaded, for the error
            message.

        Raises
        ------
        ValueError
            If the loaded graph is hierarchical.
        """
        if self.hierarchical:
            raise ValueError(
                f"{type(self).__name__} requires a flat mesh graph, "
                f"but graph '{graph_name}' is hierarchical"
            )

    @property
    def latent_spatial_dim(self) -> int:
        """
        Number of mesh nodes, where the latent variable lives.

        Returns
        -------
        int
            Number of mesh nodes.
        """
        return len(self.mesh_static_features)

    def build_learnable_prior(
        self,
        latent_dim,
        hidden_dim,
        hidden_layers,
        g2m_gnn_type,
        prior_dist,
        prior_layers,
    ):
        """
        Build the flat-graph learnable prior encoder.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step of the prior.
        prior_dist : str
            Output distribution of the prior: ``"isotropic"`` or
            ``"diagonal"``.
        prior_layers : int
            Number of on-mesh (m2m) GNN layers in the prior.

        Returns
        -------
        GraphLatentEncoder
            The learnable prior latent encoder.
        """
        return GraphLatentEncoder(
            latent_dim=latent_dim,
            g2m_edge_index=self.g2m_edge_index,
            m2m_edge_index=self.m2m_edge_index,
            hidden_dim=hidden_dim,
            m2m_layers=prior_layers,
            hidden_layers=hidden_layers,
            g2m_gnn_type=g2m_gnn_type,
            output_dist=prior_dist,
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
