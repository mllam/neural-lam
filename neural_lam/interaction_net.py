"""Interaction Network layers and helper modules used by Neural-LAM."""

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
        edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        """
        Initialise an InteractionNet message-passing layer.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge connectivity tensor in PyG format.

            * **Shape**: ``(2, num_edges)``
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
        AssertionError
            If ``aggr`` is not one of ``"sum"`` or ``"mean"``.
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

        # Make both sender and receiver indices of edge_index start at 0
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # Store number of receiver nodes according to edge_index
        self.num_rec = edge_index[1].max() + 1
        edge_index[0] = (
            edge_index[0] + self.num_rec
        )  # Make sender indices after rec
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

    def forward(self, send_rep, rec_rep, edge_rep):
        """
        Update receiver (and optionally edge) representations via message
        passing.

        Parameters
        ----------
        send_rep : torch.Tensor
            Vector representations of sender nodes.

            * **Shape**: ``(num_send, input_dim)``
        rec_rep : torch.Tensor
            Vector representations of receiver nodes.

            * **Shape**: ``(num_rec, input_dim)``
        edge_rep : torch.Tensor
            Edge representations used during message passing.

            * **Shape**: ``(num_edges, input_dim)``

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Updated receiver representations. If ``self.update_edges`` is
            ``True``, the tuple ``(rec_rep, edge_rep)`` containing the updated
            receiver and edge representations is returned.

            * **Shape**: ``(num_rec, hidden_dim)`` for receivers and
              ``(num_edges, hidden_dim)`` for edges.
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_j, x_i, edge_attr):
        """Compute messages from node ``j`` to ``i``."""
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """Aggregate messages while also returning the per-edge values."""
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        """
        Create a module that dispatches chunks of the input to separate MLPs.

        Parameters
        ----------
        mlps : Iterable[nn.Module]
            Sequence of MLPs to apply to each chunk.
        chunk_sizes : Sequence[int]
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

    def forward(self, x):
        """
        Chunk up input tensor and feed each slice through its MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to split and process.

            * **Shape**: ``(..., N, d)`` where ``N = sum(chunk_sizes)``.

        Returns
        -------
        torch.Tensor
            Concatenated MLP outputs assembled along the chunk dimension.

            * **Shape**: ``(..., N, d)``
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
