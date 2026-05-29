# Third-party
import pytest
import torch

# First-party
from neural_lam.cnn_layers import (
    FiLM2d,
    GridToNode,
    NodeToGrid,
    ResHRRRBackbone,
    ResHRRRBlock,
    SqueezeExcitation2d,
    grid_to_node,
    node_to_grid,
)
from neural_lam.datastore.base import CartesianGridShape


def test_node_to_grid_matches_datastore_flattening_order():
    batch_size = 2
    grid_shape = CartesianGridShape(x=3, y=4)
    num_channels = 5
    node_features = torch.arange(
        batch_size * grid_shape.x * grid_shape.y * num_channels,
        dtype=torch.float32,
    ).reshape(batch_size, grid_shape.x * grid_shape.y, num_channels)

    grid_features = node_to_grid(node_features, grid_shape)

    expected = node_features.reshape(
        batch_size, grid_shape.x, grid_shape.y, num_channels
    ).permute(0, 3, 1, 2)
    assert torch.equal(grid_features, expected)


def test_grid_to_node_round_trip():
    grid_shape = (3, 4)
    node_features = torch.randn(2, 12, 5)

    round_trip = grid_to_node(node_to_grid(node_features, grid_shape))

    assert torch.equal(round_trip, node_features)


def test_grid_transform_layers_round_trip():
    grid_shape = CartesianGridShape(x=2, y=3)
    node_features = torch.randn(4, 6, 7)

    grid_features = NodeToGrid(grid_shape)(node_features)
    round_trip = GridToNode(grid_shape)(grid_features)

    assert torch.equal(round_trip, node_features)


def test_node_to_grid_rejects_wrong_node_count():
    node_features = torch.randn(2, 11, 5)

    with pytest.raises(ValueError, match="node dimension"):
        node_to_grid(node_features, (3, 4))


def test_node_to_grid_rejects_non_positive_grid_shape():
    node_features = torch.randn(2, 0, 5)

    with pytest.raises(ValueError, match="positive"):
        node_to_grid(node_features, (0, 4))

    with pytest.raises(ValueError, match="positive"):
        node_to_grid(node_features, CartesianGridShape(x=0, y=4))


def test_grid_to_node_rejects_wrong_grid_shape():
    grid_features = torch.randn(2, 5, 3, 4)

    with pytest.raises(ValueError, match="spatial dimensions"):
        grid_to_node(grid_features, (4, 3))


def test_squeeze_excitation_preserves_shape():
    x = torch.randn(2, 8, 5, 4)

    y = SqueezeExcitation2d(channels=8, reduction=4)(x)

    assert y.shape == x.shape


def test_squeeze_excitation_rejects_invalid_args():
    with pytest.raises(ValueError, match="channels"):
        SqueezeExcitation2d(channels=0)

    with pytest.raises(ValueError, match="reduction"):
        SqueezeExcitation2d(channels=8, reduction=0)


def test_film2d_preserves_shape():
    x = torch.randn(2, 8, 5, 4)
    context = torch.randn(2, 3)

    y = FiLM2d(context_dim=3, channels=8)(x, context)

    assert y.shape == x.shape


def test_film2d_applies_channel_affine_conditioning():
    x = torch.ones(2, 2, 3, 4)
    context = torch.ones(2, 3)
    film = FiLM2d(context_dim=3, channels=2)
    film.proj.weight.data.zero_()
    film.proj.bias.data = torch.tensor([1.0, 2.0, 3.0, 4.0])

    y = film(x, context)

    expected = torch.empty_like(x)
    expected[:, 0] = 5.0
    expected[:, 1] = 7.0
    assert torch.equal(y, expected)


def test_film2d_rejects_shape_mismatch():
    x = torch.ones(2, 2, 3, 4)
    film = FiLM2d(context_dim=3, channels=2)

    with pytest.raises(ValueError, match="context must have shape"):
        film(x, torch.ones(2, 3, 1))

    with pytest.raises(ValueError, match="same batch size"):
        film(x, torch.ones(3, 3))

    with pytest.raises(ValueError, match="feature dimension"):
        film(x, torch.ones(2, 4))

    with pytest.raises(ValueError, match="channel dimension"):
        film(torch.ones(2, 3, 3, 4), torch.ones(2, 3))


def test_reshrrr_block_preserves_shape_without_context():
    x = torch.randn(2, 8, 5, 4)

    y = ResHRRRBlock(channels=8, reduction=4)(x)

    assert y.shape == x.shape


def test_reshrrr_block_preserves_shape_with_context():
    x = torch.randn(2, 8, 5, 4)
    context = torch.randn(2, 3)

    y = ResHRRRBlock(channels=8, reduction=4, context_dim=3)(x, context=context)

    assert y.shape == x.shape


def test_reshrrr_block_requires_context_when_film_enabled():
    x = torch.randn(2, 8, 5, 4)
    block = ResHRRRBlock(channels=8, reduction=4, context_dim=3)

    with pytest.raises(ValueError, match="context is required"):
        block(x)


def test_reshrrr_block_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        ResHRRRBlock(channels=8, kernel_size=2)


def test_reshrrr_block_rejects_non_positive_args():
    with pytest.raises(ValueError, match="channels"):
        ResHRRRBlock(channels=0)

    with pytest.raises(ValueError, match="kernel_size"):
        ResHRRRBlock(channels=8, kernel_size=0)


def test_reshrrr_backbone_returns_output_channels_without_context():
    x = torch.randn(2, 5, 6, 4)
    backbone = ResHRRRBackbone(
        input_channels=5,
        output_channels=3,
        hidden_channels=8,
        num_blocks=2,
        reduction=4,
    )

    y = backbone(x)

    assert y.shape == (2, 3, 6, 4)


def test_reshrrr_backbone_returns_output_channels_with_context():
    x = torch.randn(2, 5, 6, 4)
    context = torch.randn(2, 3)
    backbone = ResHRRRBackbone(
        input_channels=5,
        output_channels=3,
        hidden_channels=8,
        num_blocks=2,
        reduction=4,
        context_dim=3,
    )

    y = backbone(x, context=context)

    assert y.shape == (2, 3, 6, 4)


def test_reshrrr_backbone_rejects_invalid_args():
    with pytest.raises(ValueError, match="input_channels"):
        ResHRRRBackbone(input_channels=0, output_channels=3)

    with pytest.raises(ValueError, match="output_channels"):
        ResHRRRBackbone(input_channels=5, output_channels=0)

    with pytest.raises(ValueError, match="hidden_channels"):
        ResHRRRBackbone(
            input_channels=5,
            output_channels=3,
            hidden_channels=0,
        )

    with pytest.raises(ValueError, match="num_blocks"):
        ResHRRRBackbone(input_channels=5, output_channels=3, num_blocks=0)
