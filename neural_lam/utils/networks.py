"""Constructors for neural-network building blocks."""

# Third-party
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
