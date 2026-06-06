# Third-party
import torch
from torch import nn

# Local
from .datastore.base import CartesianGridShape

GridShape = CartesianGridShape | tuple[int, int]


def _grid_shape_xy(grid_shape: GridShape) -> tuple[int, int]:
    """
    Return ``(x, y)`` dimensions from a datastore or tuple grid shape.
    """
    if isinstance(grid_shape, CartesianGridShape):
        grid_x, grid_y = grid_shape.x, grid_shape.y
    else:
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain exactly two dimensions")
        grid_x, grid_y = int(grid_shape[0]), int(grid_shape[1])

    if grid_x <= 0 or grid_y <= 0:
        raise ValueError("grid_shape dimensions must be positive")

    return int(grid_x), int(grid_y)


def node_to_grid(
    node_features: torch.Tensor,
    grid_shape: GridShape,
) -> torch.Tensor:
    """
    Convert Neural-LAM node tensors to CNN grid tensors.

    Parameters
    ----------
    node_features : torch.Tensor
        Tensor of shape ``(B, N, C)``.
    grid_shape : CartesianGridShape or tuple[int, int]
        Regular grid shape as ``(x, y)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, C, x, y)``.
    """
    if node_features.ndim != 3:
        raise ValueError(
            "node_features must have shape (B, N, C), "
            f"got {tuple(node_features.shape)}"
        )

    grid_x, grid_y = _grid_shape_xy(grid_shape)
    batch_size, num_nodes, num_channels = node_features.shape
    expected_nodes = grid_x * grid_y

    if num_nodes != expected_nodes:
        raise ValueError(
            "node_features node dimension does not match grid_shape: "
            f"got {num_nodes}, expected {expected_nodes}"
        )

    return (
        node_features.reshape(batch_size, grid_x, grid_y, num_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def grid_to_node(
    grid_features: torch.Tensor,
    grid_shape: GridShape | None = None,
) -> torch.Tensor:
    """
    Convert CNN grid tensors to Neural-LAM node tensors.

    Parameters
    ----------
    grid_features : torch.Tensor
        Tensor of shape ``(B, C, x, y)``.
    grid_shape : CartesianGridShape or tuple[int, int], optional
        Expected regular grid shape as ``(x, y)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, x * y, C)``.
    """
    if grid_features.ndim != 4:
        raise ValueError(
            "grid_features must have shape (B, C, x, y), "
            f"got {tuple(grid_features.shape)}"
        )

    batch_size, num_channels, grid_x, grid_y = grid_features.shape

    if grid_shape is not None:
        expected_x, expected_y = _grid_shape_xy(grid_shape)
        if (grid_x, grid_y) != (expected_x, expected_y):
            raise ValueError(
                "grid_features spatial dimensions do not match grid_shape: "
                f"got {(grid_x, grid_y)}, expected {(expected_x, expected_y)}"
            )

    return (
        grid_features.permute(0, 2, 3, 1)
        .reshape(batch_size, grid_x * grid_y, num_channels)
        .contiguous()
    )


class NodeToGrid(nn.Module):
    """Layer wrapper for ``node_to_grid``."""

    def __init__(self, grid_shape: GridShape):
        super().__init__()
        self.grid_shape = _grid_shape_xy(grid_shape)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return node_to_grid(node_features, self.grid_shape)


class GridToNode(nn.Module):
    """Layer wrapper for ``grid_to_node``."""

    def __init__(self, grid_shape: GridShape | None = None):
        super().__init__()
        self.grid_shape = (
            None if grid_shape is None else _grid_shape_xy(grid_shape)
        )

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        return grid_to_node(grid_features, self.grid_shape)


class SqueezeExcitation2d(nn.Module):
    """Channel attention for 2D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden_channels = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class FiLM2d(nn.Module):
    """Feature-wise affine conditioning for 2D feature maps."""

    def __init__(self, context_dim: int, channels: int):
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.context_dim = context_dim
        self.channels = channels
        self.proj = nn.Linear(context_dim, 2 * channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, C, H, W), got {x.shape}")
        if context.ndim != 2:
            raise ValueError(
                "context must have shape (B, context_dim), "
                f"got {tuple(context.shape)}"
            )
        if x.shape[0] != context.shape[0]:
            raise ValueError("x and context must have the same batch size")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"x channel dimension must be {self.channels}, "
                f"got {x.shape[1]}"
            )
        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"context feature dimension must be {self.context_dim}, "
                f"got {context.shape[1]}"
            )

        gamma, beta = self.proj(context).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class ResHRRRBlock(nn.Module):
    """Residual CNN block with optional SE and FiLM conditioning."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 16,
        context_dim: int | None = None,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.se = SqueezeExcitation2d(channels, reduction=reduction)
        self.film = (
            None if context_dim is None else FiLM2d(context_dim, channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.se(x)

        if self.film is not None:
            if context is None:
                raise ValueError("context is required when FiLM is enabled")
            x = self.film(x, context)

        return self.act(x + residual)


class ResHRRRBackbone(nn.Module):
    """ResHRRR-style CNN backbone."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        kernel_size: int = 3,
        reduction: int = 16,
        context_dim: int | None = None,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.input_proj = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            [
                ResHRRRBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    reduction=reduction,
                    context_dim=context_dim,
                    padding_mode=padding_mode,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, context=context)

        return self.output_proj(x)
