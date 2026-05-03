# Standard library
from typing import Dict, Optional

# Third-party
import torch
import torch_geometric as pyg

# Local
from ....datastore import BaseDatastore
from ....interaction_net import InteractionNet
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
            One tensor per level, each of shape ``(B, N_mesh[l], d_h)``.
            Node representations at each hierarchy level. Dims: ``B`` is
            batch size, ``N_mesh[l]`` is the node count at level ``l``,
            and ``d_h`` is the hidden dimension.
        mesh_same_rep : list of torch.Tensor
            One tensor per level, each of shape ``(B, M_same[l], d_h)``.
            Same-level edge representations.
        mesh_up_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, M_up[l], d_h)``. Upward edge representations.
        mesh_down_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, M_down[l], d_h)``. Downward edge representations.

        Returns
        -------
        tuple of (list, list, list, list)
            Updated ``(mesh_rep_levels, mesh_same_rep, mesh_up_rep,
            mesh_down_rep)``.
        """

        # First join all node and edge representations to single tensors
        mesh_rep = torch.cat(mesh_rep_levels, dim=1)  # (B, N_mesh, d_h)
        mesh_edge_rep = torch.cat(
            mesh_same_rep + mesh_up_rep + mesh_down_rep, axis=1
        )  # (B, M_mesh, d_h)

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
