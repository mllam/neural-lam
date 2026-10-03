# Standard library
from pathlib import Path

# Third-party
import plotly.graph_objects as go
import pytest

# First-party
from neural_lam import plot_graph as plot_graph_module
from neural_lam import utils
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.plot_graph import (
    plot_graph,
)
from tests.dummy_datastore import DummyDatastore

REPO_ROOT = Path(__file__).resolve().parent.parent


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


class _StopAfterConfigLoad(Exception):
    """Raised by the stubbed config loader to end ``main`` early."""


@pytest.fixture
def loaded_config_path(monkeypatch):
    """Stub ``load_config_and_datastore`` and record the path it receives."""
    received = {}

    def _stub(config_path):
        received["config_path"] = config_path
        raise _StopAfterConfigLoad

    monkeypatch.setattr(plot_graph_module, "load_config_and_datastore", _stub)
    return received


def test_main_default_config_path_exists(loaded_config_path):
    """The default ``--config_path`` must point at an existing config."""
    with pytest.raises(_StopAfterConfigLoad):
        plot_graph_module.main([])

    default_path = Path(loaded_config_path["config_path"])
    assert (REPO_ROOT / default_path).is_file()
    # The default is a neural-lam config (has a `datastore` section)
    assert "datastore:" in (REPO_ROOT / default_path).read_text()


def test_main_config_path_flag(loaded_config_path):
    with pytest.raises(_StopAfterConfigLoad):
        plot_graph_module.main(["--config_path", "some/config.yaml"])
    assert loaded_config_path["config_path"] == "some/config.yaml"


def test_main_deprecated_datastore_config_path_alias(loaded_config_path):
    with pytest.warns(DeprecationWarning, match="--config_path"):
        with pytest.raises(_StopAfterConfigLoad):
            plot_graph_module.main(
                ["--datastore_config_path", "old/config.yaml"]
            )
    assert loaded_config_path["config_path"] == "old/config.yaml"


def test_main_missing_config_raises_file_not_found(tmp_path):
    """A bad path fails with a clear error rather than an argparse crash."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="does_not_exist.yaml"):
        plot_graph_module.main(["--config_path", str(missing)])


def test_main_help_lists_config_path(capsys):
    with pytest.raises(SystemExit) as exc_info:
        plot_graph_module.main(["--help"])
    assert exc_info.value.code == 0
    assert "--config_path" in capsys.readouterr().out
