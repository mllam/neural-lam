"""Sequential up/down hierarchical Neural-LAM model (Hi-LAM)."""

# Third-party
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..interaction_net import InteractionNet
from .base_hi_graph_model import BaseHiGraphModel


class HiLAM(BaseHiGraphModel):
    """
    Hierarchical graph model with message passing that goes sequentially down
    and up the hierarchy during processing.
    The Hi-LAM model from Oskarsson et al. (2023)
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        """Initialize the sequential up/down hierarchical processor."""
        super().__init__(args, config=config, datastore=datastore)

        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

    def make_same_gnns(self, args):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self, args):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self, args):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
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
        Run the downward half of the hierarchical processing sweep.

        Parameters
        ----------
        mesh_rep_levels : list[torch.Tensor]
            Mesh representations for each level.

            * **Shape**: ``(B, N_mesh[l], d_h)``
        mesh_same_rep : list[torch.Tensor]
            Same-level edge representations.

            * **Shape**: ``(B, M_same[l], d_h)``
        mesh_down_rep : list[torch.Tensor]
            Downward edge representations.

            * **Shape**: ``(B, M_down[l], d_h)``
        down_gnns : Sequence[InteractionNet]
            Message-passing networks applied to downward edges.
        same_gnns : Sequence[InteractionNet]
            Message-passing networks for same-level processing.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
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
        Run the upward half of the hierarchical processing sweep.

        Parameters
        ----------
        mesh_rep_levels : list[torch.Tensor]
            Mesh representations for each level.

            * **Shape**: ``(B, N_mesh[l], d_h)``
        mesh_same_rep : list[torch.Tensor]
            Same-level edge representations.

            * **Shape**: ``(B, M_same[l], d_h)``
        mesh_up_rep : list[torch.Tensor]
            Upward edge representations.

            * **Shape**: ``(B, M_up[l], d_h)``
        up_gnns : Sequence[InteractionNet]
            Message-passing networks applied to upward edges.
        same_gnns : Sequence[InteractionNet]
            Message-passing networks for same-level processing.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
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
        Execute one full processor iteration (down + up sweeps).

        Parameters
        ----------
        mesh_rep_levels : list[torch.Tensor]
            Mesh representations for each level.

            * **Shape**: ``(B, N_mesh[l], d_h)``
        mesh_same_rep : list[torch.Tensor]
            Same-level edge representations.

            * **Shape**: ``(B, M_same[l], d_h)``
        mesh_up_rep : list[torch.Tensor]
            Upward edge representations.

            * **Shape**: ``(B, M_up[l], d_h)``
        mesh_down_rep : list[torch.Tensor]
            Downward edge representations.

            * **Shape**: ``(B, M_down[l], d_h)``

        Returns
        -------
        tuple[
            list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
            list[torch.Tensor]
        ]
            Updated representations ``(mesh_rep_levels, mesh_same_rep,
            mesh_up_rep, mesh_down_rep)`` after both sweeps.
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
