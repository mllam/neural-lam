# Third-party
import torch

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..interaction_net import InteractionNet
from .ar_model import ARModel


class BaseGraphModel(ARModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__(args, config=config, datastore=datastore)

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        graph_dir_path = datastore.root_path / "graph" / args.graph
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.rank_zero_print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            self.g2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [args.hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            self.m2g_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute indices and define clamping functions
        self.prepare_clamping_params(config, datastore)

    def prepare_clamping_params(
        self, config: NeuralLAMConfig, datastore: BaseDatastore
    ):
        """
        Prepare parameters for clamping predicted values to valid range
        """

        # Read configs
        state_feature_names = datastore.get_vars_names(category="state")
        lower_lims = config.training.output_clamping.lower
        upper_lims = config.training.output_clamping.upper

        # Check that limits in config are for valid features
        unknown_features_lower = set(lower_lims.keys()) - set(
            state_feature_names
        )
        unknown_features_upper = set(upper_lims.keys()) - set(
            state_feature_names
        )
        if unknown_features_lower or unknown_features_upper:
            raise ValueError(
                "State feature limits were provided for unknown features: "
                f"{unknown_features_lower.union(unknown_features_upper)}"
            )

        # Constant parameters for clamping
        sigmoid_sharpness = 1
        softplus_sharpness = 1
        sigmoid_center = 0
        softplus_center = 0

        normalize_clamping_lim = (
            lambda x, feature_idx: (x - self.state_mean[feature_idx])
            / self.state_std[feature_idx]
        )

        # Check which clamping functions to use for each feature
        sigmoid_lower_upper_idx = []
        sigmoid_lower_lims = []
        sigmoid_upper_lims = []

        softplus_lower_idx = []
        softplus_lower_lims = []

        softplus_upper_idx = []
        softplus_upper_lims = []

        for feature_idx, feature in enumerate(state_feature_names):
            if feature in lower_lims and feature in upper_lims:
                assert (
                    lower_lims[feature] < upper_lims[feature]
                ), f'Invalid clamping limits for feature "{feature}",\
                     lower: {lower_lims[feature]}, larger than\
                     upper: {upper_lims[feature]}'
                sigmoid_lower_upper_idx.append(feature_idx)
                sigmoid_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
                sigmoid_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )
            elif feature in lower_lims and feature not in upper_lims:
                softplus_lower_idx.append(feature_idx)
                softplus_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
            elif feature not in lower_lims and feature in upper_lims:
                softplus_upper_idx.append(feature_idx)
                softplus_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )

        self.register_buffer(
            "sigmoid_lower_lims", torch.tensor(sigmoid_lower_lims)
        )
        self.register_buffer(
            "sigmoid_upper_lims", torch.tensor(sigmoid_upper_lims)
        )
        self.register_buffer(
            "softplus_lower_lims", torch.tensor(softplus_lower_lims)
        )
        self.register_buffer(
            "softplus_upper_lims", torch.tensor(softplus_upper_lims)
        )

        self.register_buffer(
            "clamp_lower_upper_idx", torch.tensor(sigmoid_lower_upper_idx)
        )
        self.register_buffer(
            "clamp_lower_idx", torch.tensor(softplus_lower_idx)
        )
        self.register_buffer(
            "clamp_upper_idx", torch.tensor(softplus_upper_idx)
        )

        # Define clamping functions
        self.clamp_lower_upper = lambda x: (
            self.sigmoid_lower_lims
            + (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            * torch.sigmoid(sigmoid_sharpness * (x - sigmoid_center))
        )
        self.clamp_lower = lambda x: (
            self.softplus_lower_lims
            + torch.nn.functional.softplus(
                x - softplus_center, beta=softplus_sharpness
            )
        )
        self.clamp_upper = lambda x: (
            self.softplus_upper_lims
            - torch.nn.functional.softplus(
                softplus_center - x, beta=softplus_sharpness
            )
        )

        self.inverse_clamp_lower_upper = lambda x: (
            sigmoid_center
            + utils.inverse_sigmoid(
                (x - self.sigmoid_lower_lims)
                / (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            )
            / sigmoid_sharpness
        )
        self.inverse_clamp_lower = lambda x: (
            utils.inverse_softplus(
                x - self.softplus_lower_lims, beta=softplus_sharpness
            )
            + softplus_center
        )
        self.inverse_clamp_upper = lambda x: (
            -utils.inverse_softplus(
                self.softplus_upper_lims - x, beta=softplus_sharpness
            )
            + softplus_center
        )

    def get_clamped_new_state(self, state_delta, prev_state):
        """
        Clamp prediction to valid range supplied in config
        Returns the clamped new state after adding delta to original state

        Instead of the new state being computed as
        $X_{t+1} = X_t + \\delta = X_t + model(\\{X_t,X_{t-1},...\\}, forcing)$
        The clamped values will be
        $f(f^{-1}(X_t) + model(\\{X_t, X_{t-1},... \\}, forcing))$
        Which means the model will learn to output values in the range of the
        inverse clamping function

        state_delta: (B, num_grid_nodes, feature_dim)
        prev_state: (B, num_grid_nodes, feature_dim)
        """

        # Assign new state, but overwrite clamped values of each type later
        new_state = prev_state + state_delta

        # Sigmoid/logistic clamps between ]a,b[
        if self.clamp_lower_upper_idx.numel() > 0:
            idx = self.clamp_lower_upper_idx

            new_state[:, :, idx] = self.clamp_lower_upper(
                self.inverse_clamp_lower_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]a,infty[
        if self.clamp_lower_idx.numel() > 0:
            idx = self.clamp_lower_idx

            new_state[:, :, idx] = self.clamp_lower(
                self.inverse_clamp_lower(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]-infty,b[
        if self.clamp_upper_idx.numel() > 0:
            idx = self.clamp_upper_idx

            new_state[:, :, idx] = self.clamp_upper(
                self.inverse_clamp_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        return new_state

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (num_mesh_nodes, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        batch_size = prev_state.shape[0]

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_grid_nodes, d_f)
            # NOTE: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

        # Clamp values to valid range (also add the delta to the previous state)
        new_state = self.get_clamped_new_state(rescaled_delta_mean, prev_state)

        return new_state, pred_std
