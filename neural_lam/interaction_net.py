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
        Create a new InteractionNet.

        Parameters
        ----------
        edge_index : torch.Tensor
            Shape ``(2, M)``. Edges in PyG format; both sender and receiver
            node indices start at 0. Dims: ``M`` is the number of edges.
        input_dim : int
            Dimensionality of input representations for both nodes and
            edges.
        update_edges : bool, optional
            If True, compute and return updated edge representations.
        hidden_layers : int, optional
            Number of hidden layers in each MLP.
        hidden_dim : int, optional
            Dimensionality of hidden layers. Defaults to ``input_dim``.
        edge_chunk_sizes : list of int, optional
            Chunk sizes to split edge representations into, each fed
            through a separate MLP. ``None`` means a single shared MLP.
        aggr_chunk_sizes : list of int, optional
            Chunk sizes to split aggregated node representations into,
            each fed through a separate MLP. ``None`` means a single
            shared MLP.
        aggr : str, optional
            Message aggregation method (``'sum'`` or ``'mean'``).
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

    def forward(self, send_rep, rec_rep, edge_rep):
        """
        Apply the interaction network to update receiver node
        representations, and optionally edge representations.

        Parameters
        ----------
        send_rep : torch.Tensor
            Shape ``(B, N_send, d_h)``. Sender node representations.
            Dims: ``B`` is batch size, ``N_send`` is the number of
            sender nodes, and ``d_h`` is the hidden dimension.
        rec_rep : torch.Tensor
            Shape ``(B, N_rec, d_h)``. Receiver node representations.
            Dims: ``N_rec`` is the number of receiver nodes.
        edge_rep : torch.Tensor
            Shape ``(B, M, d_h)``. Edge representations. Dims: ``M``
            is the number of edges.

        Returns
        -------
        rec_rep : torch.Tensor
            Shape ``(B, N_rec, d_h)``. Updated receiver node
            representations.
        edge_rep : torch.Tensor
            Shape ``(B, M, d_h)``. Updated edge representations.
            Only returned when ``update_edges=True``.
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
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
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
