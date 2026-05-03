# Standard library
from typing import Dict, Optional

# Third-party
from torch import nn

# Local
from ....datastore import BaseDatastore
from ....interaction_net import InteractionNet
from .hierarchical import BaseHiGraphModel


class HiLAM(BaseHiGraphModel):
    """
    Hierarchical graph model with message passing that goes sequentially down
    and up the hierarchy during processing.
    The Hi-LAM model from Oskarsson et al. (2023)
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

        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns() for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns() for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns() for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns() for _ in range(processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

    def make_same_gnns(self):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.hidden_dim,
                    hidden_layers=self.hidden_layers,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.hidden_dim,
                    hidden_layers=self.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.hidden_dim,
                    hidden_layers=self.hidden_layers,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def mesh_down_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_down_rep,
        down_gnns,
        same_gnns,
    ):
        """
        Run the downward pass of vertical processing, alternating between
        down-edges and same-level edges from the top to the bottom level.

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
        mesh_down_rep : list of torch.Tensor
            One tensor per inter-level gap, each of shape
            ``(B, M_down[l], d_h)``. Downward edge representations from
            level ``l+1`` to ``l``.
        down_gnns : nn.ModuleList
            GNNs for downward edges, one per inter-level gap.
        same_gnns : nn.ModuleList
            GNNs for same-level edges, one per level.

        Returns
        -------
        tuple of (list, list, list)
            Updated ``(mesh_rep_levels, mesh_same_rep, mesh_down_rep)``.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1]
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.num_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns
    ):
        """
        Run the upward pass of vertical processing, alternating between
        up-edges and same-level edges from the bottom to the top level.

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
            ``(B, M_up[l], d_h)``. Upward edge representations from
            level ``l`` to ``l+1``.
        up_gnns : nn.ModuleList
            GNNs for upward edges, one per inter-level gap.
        same_gnns : nn.ModuleList
            GNNs for same-level edges, one per level.

        Returns
        -------
        tuple of (list, list, list)
            Updated ``(mesh_rep_levels, mesh_same_rep, mesh_up_rep)``.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0]
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Run all processor steps (down then up at each layer depth) over
        the hierarchical mesh.

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
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_down_rep,
                down_gnns,
                down_same_gnns,
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_up_rep,
                up_gnns,
                up_same_gnns,
            )

        # NOTE: We return all, even though only down edges really are used
        # later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
