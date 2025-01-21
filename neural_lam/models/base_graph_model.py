# Standard library
from typing import Union

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

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
    ):
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
        )

        # Load graph with static features
        graph_dir_path = datastore.root_path / "graphs" / args.graph_name
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )
        for name, attr_value in graph_ldict.items():
            # NOTE: It would be good to rescale mesh node position features in
            # exactly the same way as grid node position static features.
            if name == "mesh_static_features":
                max_coord = datastore.get_xy("state").max()
                # Rescale by dividing by maximum coordinate in interior
                attr_value /= max_coord

            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        print(
            "Loaded graph with "
            f"{self.num_total_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_total_grid_nodes} grid, "
            f"{self.num_mesh_nodes} mesh)"
        )

        # interior_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for interior
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.interior_embedder = utils.make_mlp(
            [self.interior_dim] + self.mlp_blueprint_end
        )

        if self.boundary_forced:
            # Define embedder for boundary nodes
            # Optional separate embedder for boundary nodes
            if args.shared_grid_embedder:
                assert self.interior_dim == self.boundary_dim, (
                    "Grid and boundary input dimension must "
                    "be the same when using "
                    f"the same embedder, got interior_dim={self.interior_dim}, "
                    f"boundary_dim={self.boundary_dim}"
                )
                self.boundary_embedder = self.interior_embedder
            else:
                self.boundary_embedder = utils.make_mlp(
                    [self.boundary_dim] + self.mlp_blueprint_end
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
            num_rec=self.num_grid_connected_mesh_nodes,
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
            num_rec=self.num_interior_nodes,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute constants for use in time_delta encoding
        step_length_ratio = (
            datastore_boundary.step_length / datastore.step_length
        )
        min_time_delta = -(args.num_past_boundary_steps + 1) * step_length_ratio
        max_time_delta = args.num_future_boundary_steps * step_length_ratio
        time_delta_magnitude = max(max_time_delta, abs(min_time_delta))

        freq_indices = 1.0 + torch.arange(
            self.time_delta_enc_dim // 2,
            dtype=torch.float,
        )
        self.register_buffer(
            "enc_freq_denom",
            (2 * time_delta_magnitude)
            ** (2 * freq_indices / self.time_delta_enc_dim),
            persistent=False,
        )

    @property
    def num_mesh_nodes(self):
        """
        Get the total number of mesh nodes in the used mesh graph
        """
        raise NotImplementedError("num_mesh_nodes not implemented")

    @property
    def num_grid_connected_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        raise NotImplementedError(
            "num_grid_connected_mesh_nodes not implemented"
        )

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

    def predict_step(
        self, prev_state, prev_prev_state, forcing, boundary_forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        """
        batch_size = prev_state.shape[0]

        # Create full interior node features of shape
        # (B, num_interior_nodes, interior_dim)
        interior_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.interior_static_features, batch_size),
            ),
            dim=-1,
        )

        if self.boundary_forced:
            # sin-encode time deltas for boundary forcing
            boundary_forcing = self.encode_forcing_time_deltas(boundary_forcing)

            # Create full boundary node features of shape
            # (B, num_boundary_nodes, boundary_dim)
            boundary_features = torch.cat(
                (
                    boundary_forcing,
                    self.expand_to_batch(
                        self.boundary_static_features, batch_size
                    ),
                ),
                dim=-1,
            )

            # Embed boundary features
            boundary_emb = self.boundary_embedder(boundary_features)
            # (B, num_boundary_nodes, d_h)

        # Embed all features
        interior_emb = self.interior_embedder(
            interior_features
        )  # (B, num_interior_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        if self.boundary_forced:
            # Merge interior and boundary emb into input embedding
            # We enforce ordering (interior, boundary) of nodes
            full_grid_emb = torch.cat((interior_emb, boundary_emb), dim=1)
        else:
            # Only maps from interior to mesh
            full_grid_emb = interior_emb

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # Encode to mesh
        mesh_rep = self.g2m_gnn(
            full_grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = interior_emb + self.encoding_grid_mlp(
            interior_emb
        )  # (B, num_interior_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_interior_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_interior_nodes, d_grid_out)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(
                2, dim=-1
            )  # both (B, num_interior_nodes, d_f)
            # NOTE: The predicted std. is not scaled in any way here
            # linter for some reason does not think softplus is callable
            # pylint: disable-next=not-callable
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        # Rescale with one-step difference statistics
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

        # Residual connection for full state
        return prev_state + rescaled_delta_mean, pred_std

    def encode_forcing_time_deltas(self, boundary_forcing):
        """
        Build sinusoidal encodings of time deltas in boundary forcing. Removes
        original time delta features and replaces these with encoded sinusoidal
        features, returning the full new forcing tensor.

        Parameters
        ----------
        boundary_forcing : torch.Tensor
            Tensor of shape (B, num_nodes, num_forcing_dims) containing boundary
            forcing features. Time delta features are the last
            self.boundary_time_delta_dims dimensions of the num_forcing_dims
            feature dimensions.


        Returns
        -------
        encoded_forcing : torch.Tensor
            Tensor of shape (B, num_nodes, num_forcing_dims'), where the
            time delta features have been removed and encoded versions added.
            Note that this might change the number of feature dimensions.
        """
        # Extract time delta dimensions
        time_deltas = boundary_forcing[..., -self.boundary_time_delta_dims :]
        # (B, num_boundary_nodes, num_time_deltas)

        # Compute sinusoidal encodings
        frequencies = time_deltas.unsqueeze(-1) / self.enc_freq_denom
        # (B, num_boundary_nodes, num_time_deltas, num_freq)
        encodings_stacked = torch.cat(
            (
                torch.sin(frequencies),
                torch.cos(frequencies),
            ),
            dim=-1,
        )
        # (B, num_boundary_nodes, num_time_deltas, 2*num_freq)

        encoded_time_deltas = encodings_stacked.flatten(-2, -1)
        # (B, num_boundary_nodes, num_encoding_dims)

        # Put together encoded time deltas with rest of boundary_forcing
        return torch.cat(
            (
                boundary_forcing[..., : -self.boundary_time_delta_dims],
                encoded_time_deltas,
            ),
            dim=-1,
        )
