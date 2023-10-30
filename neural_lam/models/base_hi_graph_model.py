import torch
from torch import nn

from neural_lam import utils
from neural_lam.interaction_net import MeshInitNet, MeshReadOutNet
from neural_lam.models.base_graph_model import BaseGraphModel


class BaseHiGraphModel(BaseGraphModel):
    """
    Base class for hierarchical graph models.
    """

    def __init__(self, args):
        super().__init__(args)

        # Track number of nodes, edges on each level
        # Flatten lists for efficient embedding
        self.N_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.N_mesh_levels = [mesh_feat.shape[0] for mesh_feat in
                              self.mesh_static_features]  # Needs as python list for later
        N_mesh_levels_torch = torch.tensor(self.N_mesh_levels)

        # Print some useful info
        print("Loaded hierachical graph with structure:")
        for ll, N_level in enumerate(self.N_mesh_levels):
            same_level_edges = self.m2m_features[ll].shape[0]
            print(f"level {ll} - {N_level} nodes, {same_level_edges} same-level edges")

            if ll < (self.N_levels - 1):
                up_edges = self.mesh_up_features[ll].shape[0]
                down_edges = self.mesh_down_features[ll].shape[0]
                print(f"  {ll}<->{ll+1} - {up_edges} up edges, {down_edges} down edges")

        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
             for _ in range(self.N_levels)])
        self.mesh_same_embedders = nn.ModuleList(
            [utils.make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
             for _ in range(self.N_levels)])
        self.mesh_up_embedders = nn.ModuleList(
            [utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
             for _ in range(self.N_levels - 1)])
        self.mesh_down_embedders = nn.ModuleList(
            [utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
             for _ in range(self.N_levels - 1)])

        # Instantiate GNNs
        # First node index at each level
        first_index_levels = torch.cat((
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(N_mesh_levels_torch, dim=0)
        ), dim=0)
        self.register_buffer("first_index_levels", first_index_levels, persistent=False)

        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList([MeshInitNet(
            edge_index - bottom_first_index,  # Adjust
            utils.make_mlp(self.edge_mlp_blueprint),
            utils.make_mlp(self.aggr_mlp_blueprint),
            N_from_nodes, N_to_nodes)
            for edge_index, bottom_first_index, N_from_nodes, N_to_nodes in zip(
                self.mesh_up_edge_index,
                self.first_index_levels[:-1],
                self.N_mesh_levels[:-1],
                self.N_mesh_levels[1:]
        )])

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList([MeshReadOutNet(
            edge_index - bottom_first_index,  # Adjust
            utils.make_mlp(self.edge_mlp_blueprint),
            utils.make_mlp(self.aggr_mlp_blueprint),
            N_to_nodes)
            for edge_index, bottom_first_index, N_to_nodes in zip(
                self.mesh_down_edge_index,
                self.first_index_levels[:-1],  # Do not reverse order here
                self.N_mesh_levels[:-1]
        )])

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        N_mesh = sum(node_feat.shape[0] for node_feat in self.mesh_static_features)
        N_mesh_ignore = N_mesh - self.mesh_static_features[0].shape[0]
        return N_mesh, N_mesh_ignore

    def embedd_mesh_nodes(self):
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        levels_embedded = [emb(node_static_features) for emb, node_static_features in
                           zip(self.mesh_embedders, self.mesh_static_features)]
        # List of (N_level[l], d_h)

        return torch.cat(levels_embedded, dim=0)  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        batch_size = mesh_rep.shape[0]

        # - EMBEDD EDGES -
        # Embedd edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(
                emb(edge_feat),
                batch_size) for emb,
            edge_feat in zip(
                self.mesh_same_embedders,
                self.m2m_features)]
        mesh_up_rep = [
            self.expand_to_batch(
                emb(edge_feat),
                batch_size) for emb,
            edge_feat in zip(
                self.mesh_up_embedders,
                self.mesh_up_features)]
        mesh_down_rep = [
            self.expand_to_batch(
                emb(edge_feat),
                batch_size) for emb,
            edge_feat in zip(
                self.mesh_down_embedders,
                self.mesh_down_features)]

        # Create list of mesh node representations for each level,
        # each of size (B, N_mesh[l], d_h)
        mesh_rep_levels = list(torch.split(mesh_rep, self.N_mesh_levels, dim=1))

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            node_reps = torch.cat(
                (mesh_rep_levels[level_l - 1],
                 mesh_rep_levels[level_l]),
                dim=1)  # (B, N_mesh[l-1]+N_mesh[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(node_reps, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, N_mesh[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
                range(self.N_levels - 2, -1, -1),
                reversed(self.mesh_read_gnns)):
            # Extract representations
            node_reps = torch.cat(
                (mesh_rep_levels[level_l],
                 mesh_rep_levels[level_l + 1]),
                dim=1)  # (B, N_mesh[l]+N_mesh[l+1], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(node_reps, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, N_mesh[l], d_h)

        # Combine before returning
        mesh_rep = torch.cat(mesh_rep_levels, dim=1)  # (B, N_mesh, d_h)
        return mesh_rep

    def hi_processor_step(self, mesh_rep_levels, mesh_same_rep, mesh_up_rep,
                          mesh_down_rep):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        raise NotImplementedError("hi_process_step not implemented")
