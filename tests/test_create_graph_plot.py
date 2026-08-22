# Standard library
from typing import Iterator

# Third-party
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.quiver
import pytest
import torch
from torch_geometric.data import Data

# First-party
from neural_lam.create_graph import plot_graph


@pytest.fixture(scope="module", autouse=True)
def _set_agg_backend() -> None:
    """Use non-interactive backend for all plotting tests."""
    plt.switch_backend("Agg")


@pytest.fixture(autouse=True)
def close_all_figures_after_test() -> Iterator[None]:
    """Ensure test-created matplotlib figures are always cleaned up."""
    yield
    plt.close("all")


def test_directed_graph_uses_quiver_for_arrowheads() -> None:
    """Directed edges should be drawn with arrowheads, not plain lines."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]]).T
    graph = Data(edge_index=edge_index, pos=pos)

    _, axis = plot_graph(graph)

    assert axis.findobj(matplotlib.quiver.Quiver)
    assert not axis.findobj(matplotlib.collections.LineCollection)


def test_undirected_graph_uses_line_collection() -> None:
    """Undirected edges keep the existing plain-line rendering."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]]).T
    graph = Data(edge_index=edge_index, pos=pos)

    _, axis = plot_graph(graph)

    assert axis.findobj(matplotlib.collections.LineCollection)
    assert not axis.findobj(matplotlib.quiver.Quiver)
