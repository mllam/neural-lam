# Standard library
from pathlib import Path

# Third-party
import numpy as np
import plotly.graph_objects as go
import pytest
import torch

# First-party
from neural_lam import utils
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.plot_graph import (
    MESH_HEIGHT,
    plot_graph,
)
from tests.dummy_datastore import DummyDatastore


@pytest.fixture(scope="module", params=["1level", "multiscale", "hierarchical"])
def graph_fixture(request, tmp_path_factory):
    """Create a graph from a DummyDatastore and load it back.

    Parametrized over graph types: 1level (flat, keisler archetype),
    multiscale (flat multi-level, graphcast archetype) and hierarchical
    (multi-level with up/down edges).

    Returns
    -------
    tuple
        (grid_pos, hierarchical, graph_ldict, graph_name)
    """
    graph_name = request.param
    datastore = DummyDatastore()

    if graph_name == "hierarchical":
        archetype = "hierarchical"
        max_num_levels = 3
    elif graph_name == "multiscale":
        archetype = "graphcast"
        max_num_levels = 3
    elif graph_name == "1level":
        archetype = "keisler"
        max_num_levels = None
    else:
        raise ValueError(f"Unknown graph_name: {graph_name}")

    graph_dir_path = tmp_path_factory.mktemp("graph") / graph_name
    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=str(graph_dir_path),
        archetype=archetype,
        max_num_levels=max_num_levels,
    )

    grid_xy_extent = datastore.get_xy_extent(category="state")
    grid_xy_max_span = max(
        grid_xy_extent[1] - grid_xy_extent[0],
        grid_xy_extent[3] - grid_xy_extent[2],
    )

    is_hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=str(graph_dir_path),
        mesh_node_features_scaling=grid_xy_max_span,
    )

    xy = datastore.get_xy("state", stacked=True)
    grid_pos = xy / grid_xy_max_span

    return grid_pos, is_hierarchical, graph_ldict, graph_name


def test_returns_figure(graph_fixture):
    grid_pos, hierarchical, graph_ldict, graph_name = graph_fixture
    fig = plot_graph(
        grid_pos=grid_pos,
        hierarchical=hierarchical,
        graph_ldict=graph_ldict,
    )
    assert isinstance(fig, go.Figure)


def test_save_html(graph_fixture, tmp_path):
    grid_pos, hierarchical, graph_ldict, graph_name = graph_fixture
    save_path = str(tmp_path / f"graph_{graph_name}.html")
    plot_graph(
        grid_pos=grid_pos,
        hierarchical=hierarchical,
        graph_ldict=graph_ldict,
        save=save_path,
    )
    assert Path(save_path).exists()
    assert Path(save_path).stat().st_size > 0


def _minimal_graph_ldict(
    num_mesh_nodes: int, m2m_edge_index: torch.Tensor
) -> dict[str, object]:
    """Build the minimal ``load_graph``-shaped dict ``plot_graph`` needs."""
    return {
        "g2m_edge_index": torch.tensor([[0, 1], [0, 0]]),
        "m2g_edge_index": torch.tensor([[0, 0], [0, 1]]),
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": [],
        "mesh_down_edge_index": [],
        "mesh_static_features": torch.zeros(num_mesh_nodes, 2),
    }


@pytest.mark.parametrize("num_mesh_nodes", [2, 3, 5])
def test_mesh_node_without_m2m_edges(num_mesh_nodes: int) -> None:
    """Every mesh node gets a marker, even one with no m2m edge.

    ``pyg.utils.degree`` returns ``max(index) + 1`` entries unless the node
    count is passed, so a trailing mesh node with no m2m edge used to be
    missing from the z-offset array and ``np.concatenate`` raised a
    ValueError.
    """
    # Ring over nodes 0 .. num_mesh_nodes - 2, leaving the last one unconnected.
    senders = torch.arange(num_mesh_nodes - 1)
    receivers = (senders + 1) % (num_mesh_nodes - 1)
    m2m_edge_index = torch.stack((senders, receivers), dim=0)

    fig = plot_graph(
        grid_pos=np.zeros((4, 2)),
        hierarchical=False,
        graph_ldict=_minimal_graph_ldict(num_mesh_nodes, m2m_edge_index),
    )

    assert isinstance(fig, go.Figure)
    mesh_marker_z = np.asarray(fig.data[-1].z)
    assert mesh_marker_z.shape == (num_mesh_nodes,)
    # The unconnected node sits at the base mesh height, i.e. degree 0.
    assert mesh_marker_z[-1] == pytest.approx(MESH_HEIGHT)
