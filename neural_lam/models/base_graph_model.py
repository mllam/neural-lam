import torch

from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.models.ar_model import ARModel


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(self, args):
        super().__init__(args)

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first N_mesh indices,
        self.hierarchical, graph_ldict = utils.load_graph(args.graph)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.N_grid, grid_static_dim = self.grid_static_features.shape  # 63784 = 268x238
        self.N_mesh, N_mesh_ignore = self.get_num_mesh()
        if self.global_rank == 0:
            print(f"Loaded graph with {self.N_grid + self.N_mesh} nodes " +
                  f"({self.N_grid} grid, {self.N_mesh} mesh)")

        # grid_dim from data + static + batch_static
        grid_dim = 2 * self.grid_state_dim + grid_static_dim + self.grid_forcing_dim +\
            self.batch_static_feature_dim  # 2*28 + 4 + 0 + 0 =
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_embedder = utils.make_mlp([grid_dim] +
                                            self.mlp_blueprint_end)
        self.g2m_embedder = utils.make_mlp([g2m_dim] +
                                           self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] +
                                           self.mlp_blueprint_end)

        # GNNs
        # encoder
        # TODO: g2m and m2g edge indices to device?
        self.g2m_gnn = InteractionNet(
            self.g2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False)
        self.encoding_grid_mlp = utils.make_mlp([args.hidden_dim]
                                                + self.mlp_blueprint_end)

        # decoder
        self.m2g_gnn = InteractionNet(
            self.m2g_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False)

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1) + [self.grid_state_dim],
            layer_norm=False)  # No layer norm on this one

    def setup(self, stage=None):
        super().setup(stage)
        self.g2m_features = self.g2m_features.to(self.device)
        self.m2g_features = self.m2g_features.to(self.device)
        self.m2m_features = self.m2m_features.to(self.device)
        self.step_diff_mean = self.step_diff_mean.to(self.device)
        self.step_diff_std = self.step_diff_std.to(self.device)
        self.grid_static_features = self.grid_static_features.to(self.device)

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    def predict_step(self, prev_state, prev_prev_state):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, N_grid, feature_dim), X_t
        prev_prev_state: (B, N_grid, feature_dim), X_{t-1}
        batch_static_features: (B, N_grid, batch_static_feature_dim)
        forcing: (B, N_grid, forcing_dim)
        """
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (prev_state,
             prev_prev_state,
             self.expand_to_batch(
                 self.grid_static_features,
                 batch_size)),
            dim=-1)

        # Embedd all features
        grid_emb = self.grid_embedder(grid_features)  # (B, N_grid, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size)  # (B, N_mesh, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(grid_emb, mesh_emb_expanded,
                                g2m_emb_expanded)  # (B, N_mesh, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(grid_emb)  # (B, N_grid, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep,
            grid_rep,
            m2g_emb_expanded)  # (B, N_grid, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(grid_rep)  # (B, N_grid, d_f)

        # Rescale with one-step difference statistics
        rescaled_net_output = net_output * self.step_diff_std + self.step_diff_mean

        # Residual connection for full state
        return prev_state + rescaled_net_output
