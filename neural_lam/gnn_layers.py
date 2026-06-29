"""Interaction Network and PropagationNet GNN layers used by Neural-LAM."""

# Standard library
from typing import Optional, Type, Union

# Third-party
import torch
import torch_geometric as pyg
from torch import nn

# Local
from . import utils


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        edge_index: torch.Tensor,
        input_dim: int,
        update_edges: bool = True,
        hidden_layers: int = 1,
        hidden_dim: Optional[int] = None,
        edge_chunk_sizes: Optional[list[int]] = None,
        aggr_chunk_sizes: Optional[list[int]] = None,
        aggr: str = "sum",
    ) -> None:
        """
        Create a new InteractionNet.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge connectivity tensor in PyG format.
            Shape ``(2, num_edges)``.
        input_dim : int
            Dimensionality of both node and edge input representations.
        update_edges : bool, optional
            If ``True``, compute and return updated edge representations in
            addition to node representations. Default is ``True``.
        hidden_layers : int, optional
            Number of hidden layers in each MLP. Default is ``1``.
        hidden_dim : int or None, optional
            Width of hidden layers. If ``None``, defaults to ``input_dim``.
        edge_chunk_sizes : list[int] or None, optional
            Chunk sizes for splitting edge representations across separate
            MLPs. ``None`` uses a single shared MLP.
        aggr_chunk_sizes : list[int] or None, optional
            Chunk sizes for splitting aggregated node representations across
            separate MLPs. ``None`` uses a single shared MLP.
        aggr : {"sum", "mean"}, optional
            Message aggregation method. Default is ``"sum"``.

        Raises
        ------
        ValueError
            If ``aggr`` is not one of ``"sum"`` or ``"mean"``.
        """
        if aggr not in ("sum", "mean"):
            raise ValueError(f"Unknown aggregation method: {aggr}")
        super().__init__(aggr=aggr)

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

        self.num_rec = edge_index[1].max() + 1
        # edge_index is expected to be zero-based and local:
        #   edge_index[0]: sender indices in [0 .. num_snd-1]
        #   edge_index[1]: receiver indices in [0 .. num_rec-1]
        # The edge indices used in this GNN layer are defined as:
        #   receivers → [0 .. num_rec-1]
        #   senders   → [num_rec .. num_rec+num_snd-1]
        # Hence, sender indices from the input edge_index are offset
        # by num_rec to obtain the indices used in this layer.
        edge_index = torch.stack(
            (edge_index[0] + self.num_rec, edge_index[1]), dim=0
        )

        self.register_buffer("edge_index", edge_index, persistent=False)

        # Create MLPs
        edge_mlp_recipe = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1)
        aggr_mlp_recipe = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1)

        if edge_chunk_sizes is None:
            self.edge_mlp = utils.make_mlp(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [utils.make_mlp(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = utils.make_mlp(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [utils.make_mlp(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        self.update_edges = update_edges

    def forward(
        self,
        send_rep: torch.Tensor,
        rec_rep: torch.Tensor,
        edge_rep: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the interaction network to update receiver node
        representations, and optionally edge representations.

        Parameters
        ----------
        send_rep : torch.Tensor
            Sender node representations.
            Shape ``(num_send, input_dim)``.
        rec_rep : torch.Tensor
            Receiver node representations.
            Shape ``(num_rec, input_dim)``.
        edge_rep : torch.Tensor
            Edge representations.
            Shape ``(num_edges, input_dim)``.

        Returns
        -------
        rec_rep : torch.Tensor
            Updated receiver node representations.
            Shape ``(num_rec, input_dim)``.
        edge_rep : torch.Tensor
            Updated edge representations.
            Shape ``(num_edges, input_dim)``.
            Only returned when ``update_edges=True``.
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connection for receiver nodes
        rec_rep = self.node_residual_target(rec_rep, edge_rep_aggr) + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def node_residual_target(
        self, rec_rep: torch.Tensor, edge_rep_aggr: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the base tensor for the node residual connection.
        InteractionNet uses the original receiver representation.
        """
        return rec_rep

    def message(
        self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute messages from node ``j`` to ``i``."""
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    # pylint: disable-next=signature-differs
    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overridden aggregation function to:

        * return both aggregated and per-edge messages,
        * only aggregate to the number of receiver nodes (``self.num_rec``)
          rather than to ``dim_size``.
        """
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class PropagationNet(InteractionNet):
    """
    Alternative version of InteractionNet that incentivizes the propagation
    of information from sender nodes to receivers.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        edge_index: torch.Tensor,
        input_dim: int,
        update_edges: bool = True,
        hidden_layers: int = 1,
        hidden_dim: Optional[int] = None,
        edge_chunk_sizes: Optional[list[int]] = None,
        aggr_chunk_sizes: Optional[list[int]] = None,
        aggr: str = "sum",
    ) -> None:
        """Initialise the :class:`PropagationNet` layer.

        Parameters share the meaning of :class:`InteractionNet.__init__`; see
        that class for the full description. The propagation variant overrides
        ``aggr`` defaults internally to favour stability of the propagation
        residual.
        """
        # Use mean aggregation in propagation version to avoid instability
        super().__init__(
            edge_index,
            input_dim,
            update_edges=update_edges,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            edge_chunk_sizes=edge_chunk_sizes,
            aggr_chunk_sizes=aggr_chunk_sizes,
            aggr="mean",
        )

    def node_residual_target(
        self, rec_rep: torch.Tensor, edge_rep_aggr: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the base tensor for the node residual connection.
        PropagationNet uses the aggregated edge messages, propagating
        sender information to receiver nodes.
        """
        return edge_rep_aggr

    def message(
        self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages from node j to node i.
        """
        # Residual connection is to sender node, propagating information
        # to edge
        return x_j + self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))


GNN_TYPES = {
    "InteractionNet": InteractionNet,
    "PropagationNet": PropagationNet,
}


def get_gnn_class(gnn_type: str) -> Type[pyg.nn.MessagePassing]:
    """
    Look up a GNN class by name.

    Parameters
    ----------
    gnn_type : str
        One of the keys in GNN_TYPES (currently "InteractionNet" or
        "PropagationNet")

    Returns
    -------
    Type[pyg.nn.MessagePassing]
        The corresponding GNN class.
    """
    if gnn_type not in GNN_TYPES:
        raise ValueError(
            f"Unknown GNN type '{gnn_type}'. "
            f"Available types: {list(GNN_TYPES.keys())}"
        )
    return GNN_TYPES[gnn_type]


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps: list[nn.Module], chunk_sizes: list[int]) -> None:
        """
        Create a module that dispatches chunks of the input to separate MLPs.

        Parameters
        ----------
        mlps : list of nn.Module
            Sequence of MLPs to apply to each chunk.
        chunk_sizes : list of int
            Sizes used when splitting the input along ``dim=-2``.

        Raises
        ------
        AssertionError
            If the number of ``mlps`` and ``chunk_sizes`` differ.
        """
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split input along dim -2 and feed each chunk through its MLP.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., N, d)``. Input tensor where
            ``N = sum(chunk_sizes)`` and ``d`` is the feature dimension.

        Returns
        -------
        torch.Tensor
            Shape ``(..., N, d)``. Concatenated outputs from all MLPs.
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
