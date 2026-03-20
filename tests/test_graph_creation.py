# Standard library
import tempfile
from pathlib import Path
from unittest.mock import patch

# Third-party
import pytest
import torch

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.utils import BufferList
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


@pytest.mark.parametrize("graph_name", ["1level", "multiscale", "hierarchical"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_graph_creation(datastore_name, graph_name):
    """Check that the `create_ graph_from_datastore` function is implemented.

    And that the graph is created in the correct location.

    """
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

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

    required_graph_files = [
        "m2m_edge_index.pt",
        "g2m_edge_index.pt",
        "m2g_edge_index.pt",
        "m2m_features.pt",
        "g2m_features.pt",
        "m2g_features.pt",
        "mesh_features.pt",
    ]
    if hierarchical:
        required_graph_files.extend(
            [
                "mesh_up_edge_index.pt",
                "mesh_down_edge_index.pt",
                "mesh_up_features.pt",
                "mesh_down_features.pt",
            ]
        )

    # TODO: check that the number of edges is consistent over the files, for
    # now we just check the number of features
    d_features = 3
    d_mesh_static = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir_path = Path(tmpdir) / "graph" / graph_name

        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )

        assert graph_dir_path.exists()

        # check that all the required files are present
        for file_name in required_graph_files:
            assert (graph_dir_path / file_name).exists()

        # try to load each and ensure they have the right shape
        for file_name in required_graph_files:
            file_id = Path(file_name).stem  # remove the extension
            result = torch.load(graph_dir_path / file_name)

            if file_id.startswith("g2m") or file_id.startswith("m2g"):
                assert isinstance(result, torch.Tensor)

                if file_id.endswith("_index"):
                    assert (
                        result.shape[0] == 2
                    )  # adjacency matrix uses two rows
                elif file_id.endswith("_features"):
                    assert result.shape[1] == d_features

            elif file_id.startswith("m2m") or file_id.startswith("mesh"):
                assert isinstance(result, list)
                if not hierarchical:
                    assert len(result) == 1
                else:
                    if file_id.startswith("mesh_up") or file_id.startswith(
                        "mesh_down"
                    ):
                        assert len(result) == n_max_levels - 1
                    else:
                        assert len(result) == n_max_levels

                for r in result:
                    assert isinstance(r, torch.Tensor)

                    if file_id == "mesh_features":
                        assert r.shape[1] == d_mesh_static
                    elif file_id.endswith("_index"):
                        assert r.shape[0] == 2  # adjacency matrix uses two rows
                    elif file_id.endswith("_features"):
                        assert r.shape[1] == d_features


def test_hierarchical_plot_uses_correct_up_graph(tmp_path):
    """Verify that hierarchical graph visualization passes the up-graph
    (with reversed edges) to plot_graph, not the down-graph twice.

    Regression test: previously pyg_down was mistakenly passed for both
    up and down graph plots, producing identical visualizations with
    wrong in-degree coloring for the up-graph.
    """
    datastore = DummyDatastore()
    graph_dir_path = tmp_path / "graph" / "hierarchical"

    # Capture (edge_index, title) for every plot_graph call
    captured_plots = []

    def _capture_plot_graph(graph, title=None):
        captured_plots.append((graph.edge_index.clone(), title))
        return None, None

    with (
        patch(
            "neural_lam.create_graph.plot_graph",
            side_effect=_capture_plot_graph,
        ),
        patch("neural_lam.create_graph.plt"),
    ):
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=True,
            n_max_levels=3,
            create_plot=True,
        )

    up_plots = [(ei, t) for ei, t in captured_plots if t and "Up graph" in t]
    down_plots = [
        (ei, t) for ei, t in captured_plots if t and "Down graph" in t
    ]

    assert len(up_plots) > 0, "No up-graph plots were captured"
    assert len(up_plots) == len(
        down_plots
    ), "Mismatched number of up/down graph plots"

    for (up_ei, up_title), (down_ei, down_title) in zip(up_plots, down_plots):
        # Up edges must be the reverse of down edges (same nodes,
        # swapped source/target) because pyg_up is constructed by
        # inverting pyg_down.edge_index
        assert torch.equal(up_ei[0], down_ei[1]), (
            f"Up graph senders should match down graph receivers "
            f"({up_title} vs {down_title})"
        )
        assert torch.equal(up_ei[1], down_ei[0]), (
            f"Up graph receivers should match down graph senders "
            f"({up_title} vs {down_title})"
        )


class TestBufferList:
    """Tests for BufferList slice and negative index support."""

    @pytest.fixture
    def buffer_list(self):
        tensors = [torch.tensor([float(i)]) for i in range(5)]
        return BufferList(tensors)

    def test_integer_index(self, buffer_list):
        """Positive integer indexing returns the correct buffer."""
        assert torch.equal(buffer_list[0], torch.tensor([0.0]))
        assert torch.equal(buffer_list[4], torch.tensor([4.0]))

    def test_negative_index(self, buffer_list):
        """Negative indexing follows Python sequence convention."""
        assert torch.equal(buffer_list[-1], torch.tensor([4.0]))
        assert torch.equal(buffer_list[-3], torch.tensor([2.0]))

    def test_slice_full(self, buffer_list):
        """Full slice returns all buffers as a list."""
        result = buffer_list[:]
        assert len(result) == 5
        for i, tensor in enumerate(result):
            assert torch.equal(tensor, torch.tensor([float(i)]))

    def test_slice_from_index(self, buffer_list):
        """Slice from index returns the correct subset."""
        result = buffer_list[2:]
        assert len(result) == 3
        for i, tensor in enumerate(result):
            assert torch.equal(tensor, torch.tensor([float(i + 2)]))

    def test_slice_with_step(self, buffer_list):
        """Slice with step skips elements correctly."""
        result = buffer_list[::2]
        assert len(result) == 3
        expected_vals = [0.0, 2.0, 4.0]
        for tensor, expected in zip(result, expected_vals):
            assert torch.equal(tensor, torch.tensor([expected]))

    def test_slice_negative_bounds(self, buffer_list):
        """Slice with negative bounds follows Python convention."""
        result = buffer_list[-2:]
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([3.0]))
        assert torch.equal(result[1], torch.tensor([4.0]))

    def test_len(self, buffer_list):
        """Length reflects number of registered buffers."""
        assert len(buffer_list) == 5

    def test_iter(self, buffer_list):
        """Iteration yields all buffers in order."""
        values = [t.item() for t in buffer_list]
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0]
