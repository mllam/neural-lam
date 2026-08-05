# Standard library
import copy

# Third-party
import pytest
import torch
import torch_geometric as pyg

# First-party
from neural_lam.graphs.flat_weather_graph import FlatWeatherGraph

NUM_GRID = 10
NUM_MESH = 5
FEATURE_DIM = 3


def create_dummy_graph_tensors():
    """
    Create dummy tensors for instantiating a flat graph.
    In the dummy tensors all grid nodes connect to all mesh nodes in g2m and m2g
    (complete bipartite graph).
    m2m is a complete graph.
    """
    return {
        "g2m_edge_index": torch.stack(
            (
                torch.repeat_interleave(torch.arange(NUM_GRID), NUM_MESH),
                torch.arange(NUM_MESH).repeat(NUM_GRID),
            ),
            dim=0,
        ),
        "g2m_edge_features": (
            torch.zeros(NUM_GRID * NUM_MESH, FEATURE_DIM, dtype=torch.float32)
        ),
        "m2g_edge_index": torch.stack(
            (
                torch.arange(NUM_MESH).repeat(NUM_GRID),
                torch.repeat_interleave(torch.arange(NUM_GRID), NUM_MESH),
            ),
            dim=0,
        ),
        "m2g_edge_features": (
            torch.zeros(NUM_GRID * NUM_MESH, FEATURE_DIM, dtype=torch.float32)
        ),
        "m2m_edge_index": pyg.utils.remove_self_loops(
            torch.stack(
                (
                    torch.repeat_interleave(torch.arange(NUM_MESH), NUM_MESH),
                    torch.arange(NUM_MESH).repeat(NUM_MESH),
                ),
                dim=0,
            )
        )[0],
        "m2m_edge_features": (
            # Number of edges in complete graph of N nodes is N(N-1)
            torch.zeros(
                NUM_MESH * (NUM_MESH - 1), FEATURE_DIM, dtype=torch.float32
            )
        ),
        "mesh_node_features": (
            torch.zeros(NUM_MESH, FEATURE_DIM, dtype=torch.float32)
        ),
    }


def test_create_flat_graph():
    """
    Test that a Flat weather graph can be created with correct tensors
    """
    graph = FlatWeatherGraph(**create_dummy_graph_tensors())

    # Check that node counts are correct
    assert graph.num_grid_nodes == NUM_GRID, (
        "num_grid_nodes returns wrong number of grid nodes: "
        f"{graph.num_grid_nodes} (true number is {NUM_GRID})"
    )
    assert graph.num_mesh_nodes == NUM_MESH, (
        "num_mesh_nodes returns wrong number of mesh nodes: "
        f"{graph.num_mesh_nodes} (true number is {NUM_MESH})"
    )


@pytest.mark.parametrize(
    "subgraph_name,tensor_name",
    [
        (subgraph_name, tensor_name)
        for subgraph_name in ("g2m", "m2g", "m2m")
        for tensor_name in ("edge_features", "edge_index")
    ]
    + [("mesh", "node_features")],
)
def test_dtypes_flat_graph(subgraph_name, tensor_name):
    """
    Test that wrong data types properly raises errors
    """
    dummy_tensors = create_dummy_graph_tensors()

    # Test non-tensor input
    dummy_copy = copy.copy(dummy_tensors)
    dummy_copy[f"{subgraph_name}_{tensor_name}"] = 1  # Not a torch.Tensor

    with pytest.raises(AssertionError) as assertinfo:
        FlatWeatherGraph(**dummy_copy)
    assert subgraph_name in str(
        assertinfo
    ), f"AssertionError did not contain {subgraph_name}"

    # Test wrong data type
    dummy_copy = copy.copy(dummy_tensors)
    tensor_key = f"{subgraph_name}_{tensor_name}"
    dummy_copy[tensor_key] = dummy_copy[tensor_key].to(torch.float16)

    with pytest.raises(AssertionError) as assertinfo:
        FlatWeatherGraph(**dummy_copy)
    assert subgraph_name in str(
        assertinfo
    ), f"AssertionError did not contain {subgraph_name}"


@pytest.mark.parametrize("subgraph_name", ["g2m", "m2g", "m2m"])
def test_shape_match_flat_graph(subgraph_name):
    """
    Test that shape mismatch between features and edge index
    properly raises errors
    """
    dummy_tensors = create_dummy_graph_tensors()

    tensor_key = f"{subgraph_name}_edge_features"
    original_shape = dummy_tensors[tensor_key].shape
    new_shape = (original_shape[0] + 1, original_shape[1])
    dummy_tensors[tensor_key] = torch.zeros(*new_shape, dtype=torch.float32)

    with pytest.raises(AssertionError) as assertinfo:
        FlatWeatherGraph(**dummy_tensors)
    assert subgraph_name in str(
        assertinfo
    ), "AssertionError did not contain {subgraph_name}"


def test_create_graph_str_rep():
    """
    Test that string representation of graph is correct
    """
    graph = FlatWeatherGraph(**create_dummy_graph_tensors())
    str_rep = str(graph)
    # Simple test that all relevant numbers are present
    assert (
        str(NUM_GRID) in str_rep
    ), "Correct number of grid nodes not in string representation of graph"
    assert (
        str(NUM_MESH) in str_rep
    ), "Correct number of mesh nodes not in string representation of graph"

    assert (
        str(NUM_MESH * NUM_GRID) in str_rep
    ), "Correct number of g2m/m2g edges not in string representation of graph"
    assert (
        str(NUM_MESH * (NUM_MESH - 1)) in str_rep
    ), "Correct number of m2m edges not in string representation of graph"
