from torch_geometric.nn import knn_graph


def build_spherical_knn_graph(coords, k=8):
    """
    Build a KNN graph using Cartesian coordinates on the unit sphere.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing 3D coordinates.
        k (int): Number of nearest neighbors.

    Returns:
        edge_index (torch.Tensor): Graph connectivity in COO format.
    """
    return knn_graph(coords, k=k, loop=False)


def build_graph(coords, method="knn", k=8):
    """
    General graph builder interface.

    Args:
        coords (torch.Tensor): Node coordinates
        method (str): Graph construction method
        k (int): Number of neighbors

    Returns:
        edge_index (torch.Tensor)
    """
    if method == "knn":
        return build_spherical_knn_graph(coords, k)
    else:
        raise NotImplementedError(f"Graph method '{method}' not implemented")