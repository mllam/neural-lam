# Standard library
from pathlib import Path

# Third-party
import numpy as np
import plotly.graph_objects as go
import pytest

# First-party
from neural_lam import utils
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.plot_graph import (
    plot_graph,
)
from tests.dummy_datastore import DummyDatastore


@pytest.fixture(scope="module", params=["1level", "multiscale", "hierarchical"])
def graph_fixture(request, tmp_path_factory):
    """Create a graph from a DummyDatastore and load it back.

    Parametrized over graph types: 1level (flat), multiscale (flat multi-level),
    and hierarchical.

    Returns
    -------
    tuple
        (grid_pos, hierarchical, graph_ldict, graph_name)
    """
    graph_name = request.param
    datastore = DummyDatastore()

    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    elif graph_name == "multiscale":
        hierarchical = False
        n_max_levels = 3
    elif graph_name == "1level":
        hierarchical = False
        n_max_levels = 1
    else:
        raise ValueError(f"Unknown graph_name: {graph_name}")

    graph_dir_path = tmp_path_factory.mktemp("graph") / graph_name
    create_graph_from_datastore(
        datastore=datastore,
        output_root_path=str(graph_dir_path),
        hierarchical=hierarchical,
        n_max_levels=n_max_levels,
    )

    is_hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=str(graph_dir_path)
    )

    xy = datastore.get_xy("state", stacked=True)
    pos_max = np.max(np.abs(xy))
    grid_pos = xy / pos_max

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
