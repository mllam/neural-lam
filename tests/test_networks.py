# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam.utils import make_gnn_seq, make_mlp


@pytest.mark.parametrize(
    "blueprint, layer_norm, expected_layers",
    [
        ([10, 20], False, 1),  # Linear only
        ([10, 20], True, 2),  # Linear + LayerNorm
        ([10, 32, 20], False, 3),  # Linear + SiLU + Linear
        ([10, 32, 20], True, 4),  # Linear + SiLU + Linear + LayerNorm
        ([8, 16, 32, 4], True, 6),  # 2 hidden layers + LayerNorm
    ],
)
def test_make_mlp_valid_blueprint(
    blueprint: list[int], layer_norm: bool, expected_layers: int
) -> None:
    """`make_mlp` constructs an MLP with the expected layers and dimensions."""
    mlp = make_mlp(blueprint, layer_norm=layer_norm)
    assert isinstance(mlp, nn.Sequential)
    assert len(mlp) == expected_layers

    # Check forward pass
    batch_size = 4
    x = torch.randn(batch_size, blueprint[0])
    out = mlp(x)
    assert out.shape == (batch_size, blueprint[-1])

    if layer_norm:
        assert isinstance(mlp[-1], nn.LayerNorm)
        assert mlp[-1].normalized_shape == (blueprint[-1],)
    else:
        assert not isinstance(mlp[-1], nn.LayerNorm)


@pytest.mark.parametrize("invalid_blueprint", [[], [10]])
def test_make_mlp_invalid_blueprint_raises_value_error(
    invalid_blueprint: list[int],
) -> None:
    """`make_mlp` raises ValueError when blueprint has fewer than 2 entries."""
    with pytest.raises(
        ValueError, match="blueprint must contain at least 2 entries"
    ):
        make_mlp(invalid_blueprint)


@pytest.mark.parametrize("num_gnn_layers", [0, -1, -5])
def test_make_gnn_seq_invalid_layers_raises_value_error(
    num_gnn_layers: int,
) -> None:
    """`make_gnn_seq` raises ValueError when num_gnn_layers < 1."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="requires num_gnn_layers >= 1"):
        make_gnn_seq(
            edge_index=edge_index,
            num_gnn_layers=num_gnn_layers,
            hidden_layers=1,
            hidden_dim=16,
        )


@pytest.mark.parametrize("gnn_type", ["InteractionNet", "PropagationNet"])
@pytest.mark.parametrize("num_gnn_layers", [1, 3])
def test_make_gnn_seq_valid(gnn_type: str, num_gnn_layers: int) -> None:
    """`make_gnn_seq` creates a functioning Sequential GNN stack."""
    # 3 nodes: 0, 1, 2 connected in a small triangle
    edge_index = torch.tensor(
        [[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long
    )
    hidden_dim = 8
    seq = make_gnn_seq(
        edge_index=edge_index,
        num_gnn_layers=num_gnn_layers,
        hidden_layers=1,
        hidden_dim=hidden_dim,
        gnn_type=gnn_type,
    )
    assert len(seq) == num_gnn_layers

    # Test forward pass with dummy node and edge representations
    num_nodes = 3
    num_edges = edge_index.shape[1]
    node_rep = torch.randn(num_nodes, hidden_dim)
    edge_rep = torch.randn(num_edges, hidden_dim)

    out_node, out_edge = seq(node_rep, edge_rep)
    assert out_node.shape == (num_nodes, hidden_dim)
    assert out_edge.shape == (num_edges, hidden_dim)
    assert torch.isfinite(out_node).all()
    assert torch.isfinite(out_edge).all()
