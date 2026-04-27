import numpy as np
from typing import Tuple

def build_spherical_knn_graph(
    coords: np.ndarray,
    k: int,
    backend: str = "scipy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build KNN graph on sphere using cartesian coordinates.
    
    Args:
        coords: (N, 3) cartesian coordinates on unit sphere
        k: Number of neighbors per node
        backend: "scipy" (more backends to come)
    
    Returns:
        edge_index: (2, E) adjacency list
        edge_attr: (E,) distances
    """
    if backend == "scipy":
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=k+1)
        
        # Remove self-loop (first neighbor is self)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        
        n_nodes = len(coords)
        src = np.repeat(np.arange(n_nodes), k)
        dst = indices.flatten()
        edge_index = np.stack([src, dst])
        edge_attr = distances.flatten()
        
        return edge_index, edge_attr
    else:
        raise ValueError(f"Backend '{backend}' not implemented")
