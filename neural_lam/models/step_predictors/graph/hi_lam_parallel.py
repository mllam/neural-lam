"""
Parallel hierarchical graph-based LAM model.
"""

# Standard library

# Third-party
import torch
import torch_geometric as pyg

# Local
from ....datastore import BaseDatastore
from ....gnn_layers import InteractionNet
from .hierarchical import BaseHiGraphModel


class HiLAMParallel(BaseHiGraphModel):
    """
    Version of HiLAM where all message passing in the hierarchical mesh (up,
    down, inter-level) is ran in parallel.

    This is a somewhat simpler alternative to the sequential message passing
    of Hi-LAM.
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
        output_clamping_lower: dict[str, float] | None = None,
        output_clamping_upper: dict[str, float] | None = None,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
        mesh_up_gnn_type: str = "InteractionNet",
        mesh_down_gnn_type: str = "InteractionNet",
    ):
        """
        Initialize the HiLAMParallel model.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing grid metadata and data access.
        graph_name : str, default "multiscale"
            The name of the graph to load.
        hidden_dim : int, default 64
            The dimension of the hidden representations.
        hidden_layers : int, default 1
            The number of hidden layers in the MLPs.
        processor_layers : int, default 4
            The number of processor layers in the GNN.
        mesh_aggr : str, default "sum"
            The aggregation method for mesh nodes.
        num_past_forcing_steps : int, default 1
            The number of past forcing steps to include.
        num_future_forcing_steps : int, default 1
            The number of future forcing steps to include.
        output_std : bool, default False
            Whether to output a predicted standard deviation.
        output_clamping_lower : dict, optional
            Lower clamping limits for state variables.
        output_clamping_upper : dict, optional
            Upper clamping limits for state variables.
        """
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
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
            mesh_up_gnn_type=mesh_up_gnn_type,
            mesh_down_gnn_type=mesh_down_gnn_type,
        )

        # Processor GNNs
        # Create the complete edge_index combining all edges for processing
        total_edge_index_list = (
            list(self.m2m_edge_index)
            + list(self.mesh_up_edge_index)
            + list(self.mesh_down_edge_index)
        )
        total_edge_index = torch.cat(total_edge_index_list, dim=1)
        self.edge_split_sections = [ei.shape[1] for ei in total_edge_index_list]

        if processor_layers == 0:
            self.processor = lambda x, edge_attr: (x, edge_attr)
        else:
            processor_nets = [
                InteractionNet(
                    total_edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    edge_chunk_sizes=self.edge_split_sections,
                    aggr_chunk_sizes=self.level_mesh_sizes,
                )
                for _ in range(processor_layers)
            ]
            self.processor = pyg.nn.Sequential(
                "mesh_rep, edge_rep",
                [
                    (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                    for net in processor_nets
                ],
            )

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Run all processor steps in parallel across all edge types and
        hierarchy levels.

        Parameters
        ----------
        mesh_rep_levels : list of torch.Tensor
            One tensor per level, each of shape
            ``(B, num_mesh_nodes[l], hidden_dim)``. Node representations at
            each hierarchy level. Dims: ``B`` is batch size,
            ``num_mesh_nodes[l]`` is the node count at level ``l``, and
            ``hidden_dim`` is the hidden dimension.
        mesh_same_rep : list of torch.Tensor
            One tensor per level, each of shape
            ``(B, num_edges[l], hidden_dim)``. Same-level edge
            representations.
        mesh_up_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, num_edges[l], hidden_dim)``. Upward edge representations.
        mesh_down_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, num_edges[l], hidden_dim)``. Downward edge representations.

        Returns
        -------
        tuple of (list, list, list, list)
            Updated ``(mesh_rep_levels, mesh_same_rep, mesh_up_rep,
            mesh_down_rep)``.
        """

        # First join all node and edge representations to single tensors
        mesh_rep = torch.cat(
            mesh_rep_levels, dim=1
        )  # (B, num_mesh_nodes, hidden_dim)
        mesh_edge_rep = torch.cat(
            mesh_same_rep + mesh_up_rep + mesh_down_rep, axis=1
        )  # (B, num_edges, hidden_dim)

        # Here, update mesh_*_rep and mesh_rep
        mesh_rep, mesh_edge_rep = self.processor(mesh_rep, mesh_edge_rep)

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
