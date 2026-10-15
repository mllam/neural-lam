"""Constructors for neural-network building blocks."""

# Third-party
import torch_geometric as pyg
from torch import nn


def make_mlp(blueprint: list[int], layer_norm: bool = True) -> nn.Sequential:
    """
    Construct a multilayer perceptron from a blueprint of layer widths.

    Parameters
    ----------
    blueprint : list[int]
        Sequence of layer dimensions where ``blueprint[0]`` is the input size,
        ``blueprint[-1]`` is the output size, the intermediate entries specify
        the hidden layer widths, and ``len(blueprint) - 2`` is the number of
        hidden layers.
    layer_norm : bool, optional
        If ``True``, append a ``LayerNorm`` to the output as in GraphCast.

    Returns
    -------
    torch.nn.Sequential
        Sequential module implementing the specified MLP.
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def make_gnn_seq(
    edge_index,
    num_gnn_layers,
    hidden_layers,
    hidden_dim,
    gnn_type="InteractionNet",
):
    """
    Build a sequential stack of GNN layers that propagates both node and
    edge representations.

    All layer types share the ``(send, rec, edge) -> (rec, edge)``
    interface, so the stack can be applied as a single module.

    Parameters
    ----------
    edge_index : torch.Tensor
        Shape ``(2, M)``. Edge index of the edges that the GNN layers
        operate on.
    num_gnn_layers : int
        Number of stacked GNN layers; must be at least 1. Callers that
        want a no-op stage (e.g. zero intra-level layers) should skip
        building and applying the stack rather than calling this with 0.
    hidden_layers : int
        Number of hidden layers in the MLPs of each GNN layer.
    hidden_dim : int
        Dimensionality of node and edge representations.
    gnn_type : str
        GNN layer type, any key in ``gnn_layers.GNN_TYPES``.

    Returns
    -------
    pyg.nn.Sequential
        Sequential module mapping ``(mesh_rep, edge_rep)`` to updated
        ``(mesh_rep, edge_rep)``.

    Raises
    ------
    ValueError
        If ``num_gnn_layers`` is less than 1.
    """
    # First-party
    from neural_lam.gnn_layers import get_gnn_class

    if num_gnn_layers < 1:
        raise ValueError(
            "make_gnn_seq requires num_gnn_layers >= 1 "
            f"(got {num_gnn_layers}); skip the stage for a no-op."
        )
    gnn_class = get_gnn_class(gnn_type)
    return pyg.nn.Sequential(
        "mesh_rep, edge_rep",
        [
            (
                gnn_class(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                ),
                "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep",
            )
            for _ in range(num_gnn_layers)
        ],
    )
