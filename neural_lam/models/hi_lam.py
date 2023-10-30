import torch
from torch import nn

from neural_lam import utils
from neural_lam.interaction_net import InteractionNet, MeshDownNet, MeshInitNet
from neural_lam.models.base_hi_graph_model import BaseHiGraphModel


class HiLAM(BaseHiGraphModel):
    """
    Hierarchical graph model with message passing that goes sequentially down and up
    the hierarchy during processing.
    The Hi-LAM model from Oskarsson et al. (2023)
    """
    def __init__(self, args):
        super().__init__(args)

        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList([self.make_down_gnns() for _ in range(
            args.processor_layers)])  # Nested lists (proc_steps, N_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList([self.make_same_gnns() for _ in range(
            args.processor_layers)])  # Nested lists (proc_steps, N_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList([self.make_up_gnns() for _ in range(
            args.processor_layers)])  # Nested lists (proc_steps, N_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList([self.make_same_gnns() for _ in range(
            args.processor_layers)])  # Nested lists (proc_steps, N_levels)

    def make_same_gnns(self):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList([InteractionNet(
            edge_index - bottom_first_index,  # Adjust
            utils.make_mlp(self.edge_mlp_blueprint),
            utils.make_mlp(self.aggr_mlp_blueprint))
            for edge_index, bottom_first_index in zip(
                self.m2m_edge_index,
                self.first_index_levels,
        )])

    def make_up_gnns(self):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList([MeshInitNet(
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

    def make_down_gnns(self):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList([MeshDownNet(
            edge_index - bottom_first_index,  # Adjust
            utils.make_mlp(self.edge_mlp_blueprint),
            utils.make_mlp(self.aggr_mlp_blueprint),
            N_to_nodes)
            for edge_index, bottom_first_index, N_to_nodes in zip(
                self.mesh_down_edge_index,
                self.first_index_levels[:-1],  # Do not reverse order here
                self.N_mesh_levels[:-1]
        )])

    def mesh_down_step(self, mesh_rep_levels, mesh_same_rep, mesh_down_rep, down_gnns,
                       same_gnns):
        """
        Run down-part of vertical processing, sequentially alternating between processing
        using down edges and same-level edges.
        """

        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](mesh_rep_levels[-1],
                                                               mesh_same_rep[-1])

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
                range(self.N_levels - 2, -1, -1),
                reversed(down_gnns), reversed(same_gnns[:-1])):
            # Extract representations
            node_reps = torch.cat(
                (mesh_rep_levels[level_l],
                 mesh_rep_levels[level_l + 1]),
                dim=1)  # (B, N_mesh[l]+N_mesh[l+1], d_h)

            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(node_reps, down_edge_rep)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(new_node_rep,
                                                                        same_edge_rep)
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns,
                     same_gnns):
        """
        Run up-part of vertical processing, sequentially alternating between processing
        using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](mesh_rep_levels[0],
                                                            mesh_same_rep[0])

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(zip(up_gnns, same_gnns[1:]),
                                                     start=1):
            # Extract representations
            node_reps = torch.cat(
                (mesh_rep_levels[level_l - 1],
                 mesh_rep_levels[level_l]),
                dim=1)  # (B, N_mesh[l-1]+N_mesh[l], d_h)

            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(node_reps, up_edge_rep)
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(new_node_rep,
                                                                        same_edge_rep)
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

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
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
                self.mesh_down_gnns, self.mesh_down_same_gnns, self.mesh_up_gnns, self.mesh_up_same_gnns):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels, mesh_same_rep, mesh_down_rep, down_gnns,
                down_same_gnns)

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns,
                up_same_gnns)

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
