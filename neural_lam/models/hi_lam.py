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
    
    The Hi-LAM model from Oskarsson et al. (2023) implements a multi-scale 
    approach to weather forecasting using hierarchical Graph Neural Networks.
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        """
        Initializes the Hi-LAM model with hierarchical processing layers.

        Args:
            args (Namespace): Command-line arguments containing model hyperparameters 
                like hidden_dim and processor_layers.
            config (NeuralLAMConfig): Configuration object for the Neural-LAM model.
            datastore (BaseDatastore): Datastore object providing access to hierarchical 
                graph structures and edge indices.
        """
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
        Creates GNN layers for intra-level (same level) message passing.

        Args:
            args (Namespace): Model arguments specifying dimensions and layers.

        Returns:
            nn.ModuleList: A list of InteractionNet layers for each hierarchy level.
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
        Creates GNN layers for processing steps upward through the hierarchy.

        Args:
            args (Namespace): Model arguments specifying dimensions and layers.

        Returns:
            nn.ModuleList: A list of InteractionNet layers for upward edges.
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
        Creates GNN layers for processing steps downward through the hierarchy.

        Args:
            args (Namespace): Model arguments specifying dimensions and layers.

        Returns:
            nn.ModuleList: A list of InteractionNet layers for downward edges.
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
        Executes the downward part of hierarchical processing.
        
        Alternates between message passing across downward edges (inter-level) 
        and same-level edges (intra-level).

        Args:
            mesh_rep_levels (list): List of node representations at each level.
            mesh_same_rep (list): List of same-level edge representations.
            mesh_down_rep (list): List of downward edge representations.
            down_gnns (nn.ModuleList): GNN layers for downward processing.
            same_gnns (nn.ModuleList): GNN layers for same-level processing.

        Returns:
            tuple: Updated (mesh_rep_levels, mesh_same_rep, mesh_down_rep).
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

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns
    ):
        """
        Executes the upward part of hierarchical processing.

        Args:
            mesh_rep_levels (list): List of node representations.
            mesh_same_rep (list): List of same-level edge representations.
            mesh_up_rep (list): List of upward edge representations.
            up_gnns (nn.ModuleList): GNN layers for upward processing.
            same_gnns (nn.ModuleList): GNN layers for same-level processing.

        Returns:
            tuple: Updated (mesh_rep_levels, mesh_same_rep, mesh_up_rep).
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

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Main internal processor step for the hierarchical model.
        
        This step coordinates the sequence of downward and upward message 
        passing across all hierarchical levels.

        Args:
            mesh_rep_levels (list): Tensors of shape (B, N_mesh[l], d_h)
            mesh_same_rep (list): Tensors of shape (B, M_same[l], d_h)
            mesh_up_rep (list): Tensors of shape (B, M_up[l -> l+1], d_h)
            mesh_down_rep (list): Tensors of shape (B, M_down[l <- l+1], d_h)

        Returns:
            tuple: Updated representation lists for levels, same, up, and down edges.
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

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep