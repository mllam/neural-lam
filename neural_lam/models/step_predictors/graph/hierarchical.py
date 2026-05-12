# Standard library
from typing import Dict, Optional

# Third-party
from torch import nn

# Local
from .... import utils
from ....datastore import BaseDatastore
from ....interaction_net import InteractionNet
from .base import BaseGraphModel


class BaseHiGraphModel(BaseGraphModel):
    """
    Base class for hierarchical graph models.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        graph_name: str = "multiscale",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        processor_layers: int = 4,
        mesh_aggr: str = "sum",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            processor_layers=processor_layers,
            mesh_aggr=mesh_aggr,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        # Track number of nodes, edges on each level
        # Flatten lists for efficient embedding
        self.num_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]  # Needs as python list for later

        # Print some useful info
        utils.log_on_rank_zero("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(self.level_mesh_sizes):
            same_level_edges = self.m2m_features[level_index].shape[0]
            utils.log_on_rank_zero(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (self.num_levels - 1):
                up_edges = self.mesh_up_features[level_index].shape[0]
                down_edges = self.mesh_down_features[level_index].shape[0]
                utils.log_on_rank_zero(f"  {level_index}<->{level_index + 1}")
                utils.log_on_rank_zero(
                    f" - {up_edges} up edges, {down_edges} down edges"
                )
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_same_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_up_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )
        self.mesh_down_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        num_mesh_nodes = sum(
            node_feat.shape[0] for node_feat in self.mesh_static_features
        )
        num_mesh_nodes_ignore = (
            num_mesh_nodes - self.mesh_static_features[0].shape[0]
        )
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self):
        """
        Embed static mesh node features for the bottom level only;
        remaining levels are embedded at the start of ``process_step``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_mesh_nodes[0], d_h)``. Embedded bottom-level
            mesh node representations. Dims: ``num_mesh_nodes[0]`` is
            the number of nodes at level 0 and ``d_h`` is the hidden
            dimension.
        """
        return self.mesh_embedders[0](self.mesh_static_features[0])

    def process_step(self, mesh_rep):
        """
        Process the mesh representation across all hierarchy levels,
        implementing the full init-process-readout cycle.

        Parameters
        ----------
        mesh_rep : torch.Tensor
            Shape ``(B, num_mesh_nodes[0], d_h)``. Bottom-level mesh
            node representations from the encoder. Dims: ``B`` is batch
            size, ``num_mesh_nodes[0]`` is the number of nodes at
            level 0, and ``d_h`` is the hidden dimension.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes[0], d_h)``. Updated bottom-level
            mesh node representations. Dims: same as ``mesh_rep``.
        """
        batch_size = mesh_rep.shape[0]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(emb(node_static_features), batch_size)
            for emb, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(self.mesh_static_features)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_same_embedders, self.m2m_features
            )
        ]
        mesh_up_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        mesh_down_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(
                send_node_rep, rec_node_rep, edge_rep
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = (
                new_node_rep  # (B, num_mesh_nodes[l], d_h)
            )
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.num_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = (
                new_node_rep  # (B, num_mesh_nodes[l], d_h)
            )

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Internal processor step between mesh init and read-out.

        Parameters
        ----------
        mesh_rep_levels : list of torch.Tensor
            One tensor per level, each of shape
            ``(B, num_mesh_nodes[l], d_h)``. Node representations at
            each hierarchy level. Dims: ``B`` is batch size,
            ``num_mesh_nodes[l]`` is the node count at level ``l``, and
            ``d_h`` is the hidden dimension.
        mesh_same_rep : list of torch.Tensor
            One tensor per level, each of shape ``(B, M_same[l], d_h)``.
            Same-level edge representations. ``M_same[l]`` is the edge
            count at level ``l``.
        mesh_up_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, M_up[l], d_h)``. Upward edge representations from
            level ``l`` to ``l+1``.
        mesh_down_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, M_down[l], d_h)``. Downward edge representations from
            level ``l+1`` to ``l``.

        Returns
        -------
        tuple of (list, list, list, list)
            Updated ``(mesh_rep_levels, mesh_same_rep, mesh_up_rep,
            mesh_down_rep)`` in the same order as the inputs.
        """
        raise NotImplementedError("hi_process_step not implemented")
