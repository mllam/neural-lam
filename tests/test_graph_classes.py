# Standard library
import copy

# Third-party
import pytest
import torch

# First-party
from neural_lam.graphs.flat_weather_graph import FlatWeatherGraph


def create_dummy_graph_tensors():
    """
    Create dummy tensors for instantiating a flat graph
    """
    num_grid = 10
    num_mesh = 5
    feature_dim = 3

    return {
        "g2m_edge_index": torch.zeros(2, num_grid, dtype=torch.long),
        "g2m_edge_features": (
            torch.zeros(num_grid, feature_dim, dtype=torch.float32)
        ),
        "m2g_edge_index": torch.zeros(2, num_grid, dtype=torch.long),
        "m2g_edge_features": (
            torch.zeros(num_grid, feature_dim, dtype=torch.float32)
        ),
        "m2m_edge_index": torch.zeros(2, num_mesh, dtype=torch.long),
        "m2m_edge_features": (
            torch.zeros(num_mesh, feature_dim, dtype=torch.float32)
        ),
        "mesh_node_features": (
            torch.zeros(num_mesh, feature_dim, dtype=torch.float32)
        ),
    }


def test_create_flat_graph():
    """
    Test that a Flat weather graph can be created with correct tensors
    """
    FlatWeatherGraph(**create_dummy_graph_tensors())


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
    ), "AssertionError did not contain {subgraph_name}"

    # Test wrong data type
    dummy_copy = copy.copy(dummy_tensors)
    tensor_key = f"{subgraph_name}_{tensor_name}"
    dummy_copy[tensor_key] = dummy_copy[tensor_key].to(torch.float16)

    with pytest.raises(AssertionError) as assertinfo:
        FlatWeatherGraph(**dummy_copy)
    assert subgraph_name in str(
        assertinfo
    ), "AssertionError did not contain {subgraph_name}"


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
