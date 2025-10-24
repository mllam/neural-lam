# Standard library
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

# Third-party
import torch
from torch import Tensor


def _ensure_tuple_of_tensors(value) -> Tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, Tensor) for item in value):
            raise TypeError("Expected all elements to be torch.Tensor instances")
        return tuple(value)
    raise TypeError("Expected a tensor or an iterable of tensors")


@dataclass(frozen=True)
class GraphEdgesAndFeatures:
    """
    Static adjacency and feature tensors describing the graph structure.

    Attributes
    ----------
    Always present
    -   hierarchical : bool
            Whether the graph contains multiple mesh levels.
    -   g2m_edge_index : torch.Tensor
            COO indices describing grid-to-mesh edges (shape ``[2, E_g2m]``).
    -   m2g_edge_index : torch.Tensor
            COO indices describing mesh-to-grid edges (shape ``[2, E_m2g]``).
    -   m2m_edge_index : tuple[torch.Tensor, ...]
            Intra-level mesh-to-mesh edges per level (hierarchical) or a single
            tensor when non-hierarchical.
    -   g2m_features : torch.Tensor
            Feature matrix for grid-to-mesh edges (shape ``[E_g2m, F_g2m]``).
    -   m2g_features : torch.Tensor
            Feature matrix for mesh-to-grid edges (shape ``[E_m2g, F_m2g]``).
    -   m2m_features : tuple[torch.Tensor, ...]
            Feature matrices for intra-level mesh edges.
    -   mesh_static_features : tuple[torch.Tensor, ...]
            Static mesh node features per level.
    Only for hierarchical graphs
        mesh_up_edge_index : tuple[torch.Tensor, ...]
            Edge indices connecting level ``l`` to ``l+1``.
        mesh_down_edge_index : tuple[torch.Tensor, ...]
            Edge indices connecting level ``l+1`` back to ``l``.
        mesh_up_features : tuple[torch.Tensor, ...]
            Feature matrices associated with upward inter-level edges.
        mesh_down_features : tuple[torch.Tensor, ...]
            Feature matrices associated with downward inter-level edges.
    """

    hierarchical: bool
    g2m_edge_index: Tensor
    m2g_edge_index: Tensor
    m2m_edge_index: Tuple[Tensor, ...] = field(default_factory=tuple)
    mesh_up_edge_index: Tuple[Tensor, ...] = field(default_factory=tuple)
    mesh_down_edge_index: Tuple[Tensor, ...] = field(default_factory=tuple)
    g2m_features: Tensor = field(default=None)
    m2g_features: Tensor = field(default=None)
    m2m_features: Tuple[Tensor, ...] = field(default_factory=tuple)
    mesh_up_features: Tuple[Tensor, ...] = field(default_factory=tuple)
    mesh_down_features: Tuple[Tensor, ...] = field(default_factory=tuple)
    mesh_static_features: Tuple[Tensor, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.g2m_edge_index, Tensor):
            raise TypeError("g2m_edge_index must be a torch.Tensor")
        if not isinstance(self.m2g_edge_index, Tensor):
            raise TypeError("m2g_edge_index must be a torch.Tensor")
        if self.g2m_features is None or self.m2g_features is None:
            raise ValueError("Edge feature tensors must be provided")
        if not isinstance(self.g2m_features, Tensor) or not isinstance(
            self.m2g_features, Tensor
        ):
            raise TypeError("Edge feature tensors must be torch.Tensor instances")

        object.__setattr__(
            self, "m2m_edge_index", _ensure_tuple_of_tensors(self.m2m_edge_index)
        )
        object.__setattr__(
            self, "m2m_features", _ensure_tuple_of_tensors(self.m2m_features)
        )
        object.__setattr__(
            self,
            "mesh_static_features",
            _ensure_tuple_of_tensors(self.mesh_static_features),
        )
        object.__setattr__(
            self,
            "mesh_up_edge_index",
            _ensure_tuple_of_tensors(self.mesh_up_edge_index),
        )
        object.__setattr__(
            self,
            "mesh_down_edge_index",
            _ensure_tuple_of_tensors(self.mesh_down_edge_index),
        )
        object.__setattr__(
            self,
            "mesh_up_features",
            _ensure_tuple_of_tensors(self.mesh_up_features),
        )
        object.__setattr__(
            self,
            "mesh_down_features",
            _ensure_tuple_of_tensors(self.mesh_down_features),
        )

        # Validation of hierarchy consistency
        num_levels = len(self.mesh_static_features)
        if num_levels == 0:
            raise ValueError("mesh_static_features must contain at least one level")

        if len(self.m2m_edge_index) != num_levels:
            raise ValueError(
                "m2m_edge_index must match number of mesh levels "
                f"(expected {num_levels}, got {len(self.m2m_edge_index)})"
            )
        if len(self.m2m_features) != num_levels:
            raise ValueError(
                "m2m_features must match number of mesh levels "
                f"(expected {num_levels}, got {len(self.m2m_features)})"
            )

        if self.hierarchical:
            if num_levels < 2:
                raise ValueError(
                    "Hierarchical graphs require at least two mesh levels"
                )
            expected = num_levels - 1
            for name, values in (
                ("mesh_up_edge_index", self.mesh_up_edge_index),
                ("mesh_down_edge_index", self.mesh_down_edge_index),
                ("mesh_up_features", self.mesh_up_features),
                ("mesh_down_features", self.mesh_down_features),
            ):
                if len(values) != expected:
                    raise ValueError(
                        f"{name} must contain {expected} entries for hierarchical graphs "
                        f"(got {len(values)})"
                    )
        else:
            for name, values in (
                ("mesh_up_edge_index", self.mesh_up_edge_index),
                ("mesh_down_edge_index", self.mesh_down_edge_index),
                ("mesh_up_features", self.mesh_up_features),
                ("mesh_down_features", self.mesh_down_features),
            ):
                if len(values) != 0:
                    raise ValueError(
                        f"{name} must be empty for non-hierarchical graphs"
                    )

    @property
    def num_levels(self) -> int:
        """Number of mesh levels represented by this graph."""
        return len(self.mesh_static_features)

    def as_batch_dict(self) -> Dict[str, object]:
        """
        Convert the edges/features into a dictionary suitable for batching.
        """

        def _hier_or_first(values: Tuple[Tensor, ...]) -> object:
            if self.hierarchical:
                return [tensor for tensor in values]
            return values[0]

        def _hier_or_empty(values: Tuple[Tensor, ...]) -> object:
            if self.hierarchical:
                return [tensor for tensor in values]
            return []

        return {
            "hierarchical": self.hierarchical,
            "g2m_edge_index": self.g2m_edge_index,
            "m2g_edge_index": self.m2g_edge_index,
            "m2m_edge_index": _hier_or_first(self.m2m_edge_index),
            "mesh_up_edge_index": _hier_or_empty(self.mesh_up_edge_index),
            "mesh_down_edge_index": _hier_or_empty(self.mesh_down_edge_index),
            "g2m_features": self.g2m_features,
            "m2g_features": self.m2g_features,
            "m2m_features": _hier_or_first(self.m2m_features),
            "mesh_up_features": _hier_or_empty(self.mesh_up_features),
            "mesh_down_features": _hier_or_empty(self.mesh_down_features),
            "mesh_static_features": _hier_or_first(self.mesh_static_features),
        }


def _to_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(values, tuple):
        return values
    return tuple(values)


@dataclass(frozen=True)
class GraphSizes:
    """
    Static metadata describing dimensionality of graph tensors.

    Attributes
    ----------
    Required for all graphs
        hierarchical : bool
            Whether the graph contains multiple mesh levels.
        num_mesh_nodes : int
            Total number of mesh nodes across all levels.
        g2m_dim : int
            Dimensionality of features on grid-to-mesh edges.
        m2g_dim : int
            Dimensionality of features on mesh-to-grid edges.
        mesh_level_sizes : tuple[int, ...]
            Number of mesh nodes at each level (ordered bottom to top).
        mesh_feature_dims : tuple[int, ...]
            Static feature dimensionality for mesh nodes on each level.
        m2m_feature_dims : tuple[int, ...]
            Feature dimensionality of same-level mesh-to-mesh edges.
        m2m_edge_counts : tuple[int, ...]
            Number of intra-level mesh edges for each level.

    Only for hierarchical graphs
        mesh_up_feature_dims : tuple[int, ...]
            Feature dimensionality of edges connecting level ``l`` to ``l+1``.
        mesh_up_edge_counts : tuple[int, ...]
            Number of upward inter-level edges per level pair.
        mesh_down_feature_dims : tuple[int, ...]
            Feature dimensionality of edges connecting level ``l+1`` back to ``l``.
        mesh_down_edge_counts : tuple[int, ...]
            Number of downward inter-level edges per level pair.
    """

    hierarchical: bool
    num_mesh_nodes: int
    g2m_dim: int
    m2g_dim: int
    mesh_level_sizes: Tuple[int, ...] = field(default_factory=tuple)
    mesh_feature_dims: Tuple[int, ...] = field(default_factory=tuple)
    m2m_feature_dims: Tuple[int, ...] = field(default_factory=tuple)
    m2m_edge_counts: Tuple[int, ...] = field(default_factory=tuple)
    mesh_up_feature_dims: Tuple[int, ...] = field(default_factory=tuple)
    mesh_up_edge_counts: Tuple[int, ...] = field(default_factory=tuple)
    mesh_down_feature_dims: Tuple[int, ...] = field(default_factory=tuple)
    mesh_down_edge_counts: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "mesh_level_sizes", _to_tuple(self.mesh_level_sizes))
        object.__setattr__(self, "mesh_feature_dims", _to_tuple(self.mesh_feature_dims))
        object.__setattr__(
            self, "m2m_feature_dims", _to_tuple(self.m2m_feature_dims)
        )
        object.__setattr__(self, "m2m_edge_counts", _to_tuple(self.m2m_edge_counts))
        object.__setattr__(
            self, "mesh_up_feature_dims", _to_tuple(self.mesh_up_feature_dims)
        )
        object.__setattr__(
            self, "mesh_up_edge_counts", _to_tuple(self.mesh_up_edge_counts)
        )
        object.__setattr__(
            self, "mesh_down_feature_dims", _to_tuple(self.mesh_down_feature_dims)
        )
        object.__setattr__(
            self, "mesh_down_edge_counts", _to_tuple(self.mesh_down_edge_counts)
        )

        if self.num_mesh_nodes <= 0:
            raise ValueError("num_mesh_nodes must be positive")
        if self.g2m_dim <= 0:
            raise ValueError("g2m_dim must be positive")
        if self.m2g_dim <= 0:
            raise ValueError("m2g_dim must be positive")

        if not self.mesh_level_sizes:
            raise ValueError("mesh_level_sizes must contain at least one level size")

        expected_sum = sum(self.mesh_level_sizes)
        if expected_sum != self.num_mesh_nodes:
            raise ValueError(
                "mesh_level_sizes must sum to num_mesh_nodes "
                f"(got {expected_sum} vs {self.num_mesh_nodes})"
            )

        num_levels = len(self.mesh_level_sizes)
        for name, values in (
            ("mesh_feature_dims", self.mesh_feature_dims),
            ("m2m_feature_dims", self.m2m_feature_dims),
            ("m2m_edge_counts", self.m2m_edge_counts),
        ):
            if len(values) != num_levels:
                raise ValueError(
                    f"{name} must have the same length as mesh_level_sizes "
                    f"(expected {num_levels}, got {len(values)})"
                )

        if self.hierarchical:
            if num_levels < 2:
                raise ValueError(
                    "Hierarchical graphs must have at least two mesh levels"
                )
            expected_edges = num_levels - 1
            for name, values in (
                ("mesh_up_feature_dims", self.mesh_up_feature_dims),
                ("mesh_up_edge_counts", self.mesh_up_edge_counts),
                ("mesh_down_feature_dims", self.mesh_down_feature_dims),
                ("mesh_down_edge_counts", self.mesh_down_edge_counts),
            ):
                if len(values) != expected_edges:
                    raise ValueError(
                        f"{name} must have length {expected_edges} for "
                        f"hierarchical graphs (got {len(values)})"
                    )
        else:
            for name, values in (
                ("mesh_up_feature_dims", self.mesh_up_feature_dims),
                ("mesh_up_edge_counts", self.mesh_up_edge_counts),
                ("mesh_down_feature_dims", self.mesh_down_feature_dims),
                ("mesh_down_edge_counts", self.mesh_down_edge_counts),
            ):
                if values:
                    raise ValueError(
                        f"{name} must be empty for non-hierarchical graphs"
                    )

    @property
    def num_levels(self) -> int:
        """Number of mesh levels represented by this metadata."""
        return len(self.mesh_level_sizes)
