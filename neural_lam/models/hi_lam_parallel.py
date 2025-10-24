# Third-party
import torch
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..graph_meta import GraphSizes
from ..interaction_net import InteractionNet
from .base_hi_graph_model import BaseHiGraphModel


class HiLAMParallel(BaseHiGraphModel):
    """
    Version of HiLAM where all message passing in the hierarchical mesh (up,
    down, inter-level) is ran in parallel.

    This is a somewhat simpler alternative to the sequential message passing
    of Hi-LAM.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_sizes: GraphSizes,
    ):
        """
        Parameters
        ----------
        graph_sizes : GraphSizes
            Graph metadata adhering to the ``BaseGraphModel`` specification,
            containing mesh level sizes, feature dimensions, and edge counts used
            to construct the parallel processor GNNs.
        """
        super().__init__(
            args,
            config=config,
            datastore=datastore,
            graph_sizes=graph_sizes,
        )

        # Processor GNNs
        self.edge_split_sections = (
            list(self.graph_sizes.m2m_edge_counts)
            + list(self.graph_sizes.mesh_up_edge_counts)
            + list(self.graph_sizes.mesh_down_edge_counts)
        )

        if args.processor_layers == 0:
            self.processor_nets = None
        else:
            self.processor_nets = nn.ModuleList(
                [
                    InteractionNet(
                        args.hidden_dim,
                        hidden_layers=args.hidden_layers,
                        edge_chunk_sizes=self.edge_split_sections,
                        aggr_chunk_sizes=self.level_mesh_sizes,
                    )
                    for _ in range(args.processor_layers)
                ]
            )
        self.current_processor_edge_index = None

    def set_graph(self, graph):
        super().set_graph(graph)
        graph_edges = (
            list(self.current_graph["m2m_edge_index"])
            + list(self.current_graph["mesh_up_edge_index"])
            + list(self.current_graph["mesh_down_edge_index"])
        )
        if graph_edges:
            self.current_processor_edge_index = torch.cat(graph_edges, dim=1)
        else:
            self.current_processor_edge_index = None

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
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
        mesh_rep = torch.cat(mesh_rep_levels, dim=1)  # (B, N_mesh, d_h)
        mesh_edge_rep = torch.cat(
            mesh_same_rep + mesh_up_rep + mesh_down_rep, axis=1
        )  # (B, M_mesh, d_h)

        # Here, update mesh_*_rep and mesh_rep
        if self.processor_nets is not None:
            if self.current_processor_edge_index is None:
                raise RuntimeError(
                    "Processor edge index not set for current graph."
                )
            for net in self.processor_nets:
                mesh_rep, mesh_edge_rep = net(
                    mesh_rep,
                    mesh_rep,
                    mesh_edge_rep,
                    self.current_processor_edge_index,
                )

        # Split up again for read-out step
        mesh_rep_levels = list(
            torch.split(mesh_rep, self.level_mesh_sizes, dim=1)
        )
        mesh_edge_rep_sections = torch.split(
            mesh_edge_rep, self.edge_split_sections, dim=1
        )

        mesh_same_rep = mesh_edge_rep_sections[: self.num_levels]
        mesh_up_rep = mesh_edge_rep_sections[
            self.num_levels : self.num_levels + (self.num_levels - 1)
        ]
        mesh_down_rep = mesh_edge_rep_sections[
            self.num_levels + (self.num_levels - 1) :
        ]  # Last are down edges

        # TODO: We return all, even though only down edges really are used
        # later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
