import torch
import torch_geometric as pyg

from neural_lam import utils
from neural_lam.models.base_hi_graph_model import BaseHiGraphModel
from neural_lam.interaction_net import HiInteractionNet

class HiLAMParallel(BaseHiGraphModel):
    """
    Version of HiLAM where all message passing in the hierarchical mesh (up, down,
    inter-level) is ran in paralell.

    This is a somewhat simpler alternative to the sequential message passing of Hi-LAM.
    """
    def __init__(self, args):
        super().__init__(args)

        # Processor GNNs
        # Create the complete total edge_index combining all edges for processing
        total_edge_index_list = list(self.m2m_edge_index) +\
                list(self.mesh_up_edge_index) + list(self.mesh_down_edge_index)
        total_edge_index = torch.cat(total_edge_index_list, dim=1)
        self.edge_split_sections = [ei.shape[1] for ei in total_edge_index_list]

        if args.processor_layers == 0:
            self.processor = (lambda x, edge_attr: (x, edge_attr))
        else:
            processor_nets = [HiInteractionNet(total_edge_index,
                    [utils.make_mlp(self.edge_mlp_blueprint) for _ in
                        range(len(self.edge_split_sections))],
                    [utils.make_mlp(self.aggr_mlp_blueprint) for _ in
                        range(self.N_levels)],
                    self.edge_split_sections, self.N_mesh_levels, aggr=args.mesh_aggr)
                for _ in range(args.processor_layers)]
            self.processor = pyg.nn.Sequential("x, edge_attr", [
                    (net, "x, edge_attr -> x, edge_attr")
                for net in processor_nets])

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

        # First join all node and edge representations to single tensors
        mesh_rep = torch.cat(mesh_rep_levels, dim=1) # (B, N_mesh, d_h)
        mesh_edge_rep = torch.cat(mesh_same_rep + mesh_up_rep + mesh_down_rep,
                axis=1) # (B, M_mesh, d_h)

        # Here, update mesh_*_rep and mesh_rep
        mesh_rep, mesh_edge_rep = self.processor(mesh_rep, mesh_edge_rep)

        # Split up again for read-out step
        mesh_rep_levels = list(torch.split(mesh_rep, self.N_mesh_levels, dim=1))
        mesh_edge_rep_sections = torch.split(mesh_edge_rep, self.edge_split_sections,
                dim=1)

        mesh_same_rep = mesh_edge_rep_sections[:self.N_levels]
        mesh_up_rep = mesh_edge_rep_sections[
                self.N_levels:self.N_levels+(self.N_levels-1)]
        mesh_down_rep = mesh_edge_rep_sections[
                self.N_levels+(self.N_levels-1):] # Last are down edges

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
