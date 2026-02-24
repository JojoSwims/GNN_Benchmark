"""Graph utility functions for adjacency matrix operations."""

from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Uses the haversine formula with the mean Earth radius.

    Args:
        lat1: Latitude of first point (degrees).
        lon1: Longitude of first point (degrees).
        lat2: Latitude of second point (degrees).
        lon2: Longitude of second point (degrees).

    Returns:
        Distance in meters.
    """
    R = 6371008.8  # Mean Earth radius in meters

    phi1, lambda1, phi2, lambda2 = map(radians, (lat1, lon1, lat2, lon2))

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


def adjacency_from_edges(
    edges: pd.DataFrame,
    nodes: list[str],
    symmetric: bool = True,
    default_diagonal: float = 0.0,
) -> np.ndarray:
    """
    Convert edges DataFrame to adjacency matrix.

    Args:
        edges: DataFrame with columns [src, dst, cost]
        nodes: Ordered list of node IDs
        symmetric: If True, make matrix symmetric by averaging
        default_diagonal: Value for diagonal entries

    Returns:
        NxN numpy array
    """
    N = len(nodes)
    node_to_idx = {str(n): i for i, n in enumerate(nodes)}

    adj = np.zeros((N, N), dtype=np.float32)
    np.fill_diagonal(adj, default_diagonal)

    for _, row in edges.iterrows():
        src_idx = node_to_idx.get(str(row["src"]))
        dst_idx = node_to_idx.get(str(row["dst"]))

        if src_idx is not None and dst_idx is not None:
            adj[src_idx, dst_idx] = row["cost"]

    if symmetric:
        # Symmetrize by averaging
        adj = (adj + adj.T) / 2

    return adj


def normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Symmetric normalization: D^(-1/2) * A * D^(-1/2)

    Args:
        adj: Adjacency matrix
        add_self_loops: If True, add self-loops before normalizing

    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        adj = adj + np.eye(adj.shape[0])

    # Compute degree matrix
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0

    # D^(-1/2) * A * D^(-1/2)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def random_walk_matrix(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Random walk normalization: D^(-1) * A

    Args:
        adj: Adjacency matrix
        add_self_loops: If True, add self-loops before normalizing

    Returns:
        Random walk normalized matrix
    """
    if add_self_loops:
        adj = adj + np.eye(adj.shape[0])

    # Compute degree matrix
    d = np.sum(adj, axis=1)
    d_inv = np.power(d, -1.0, where=d > 0)
    d_inv[d == 0] = 0

    # D^(-1) * A
    d_mat_inv = np.diag(d_inv)
    return d_mat_inv @ adj


def normalized_laplacian(adj: np.ndarray, add_self_loops: bool = False) -> np.ndarray:
    """
    Compute normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)

    Args:
        adj: Adjacency matrix
        add_self_loops: If True, add self-loops before computing

    Returns:
        Normalized Laplacian matrix
    """
    N = adj.shape[0]
    norm_adj = normalize_adjacency(adj, add_self_loops=add_self_loops)
    return np.eye(N) - norm_adj


def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Compute scaled Laplacian: 2 * L / lambda_max - I

    Used in ChebNet and related spectral methods.

    Args:
        adj: Adjacency matrix

    Returns:
        Scaled Laplacian matrix
    """
    N = adj.shape[0]
    L = normalized_laplacian(adj)

    # Compute largest eigenvalue
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_max = eigenvalues[-1]

    if lambda_max == 0:
        lambda_max = 2.0

    return 2.0 * L / lambda_max - np.eye(N)


def exp_decay_adjacency(
    edges: pd.DataFrame,
    nodes: list[str],
    sigma: float = 0.1,
    epsilon: float = 0.5,
) -> np.ndarray:
    """
    Create adjacency matrix with exponential decay weights.

    w_ij = exp(-d_ij^2 / sigma^2) if exp(...) >= epsilon else 0

    Commonly used for traffic datasets where edge costs are distances.

    Args:
        edges: DataFrame with columns [src, dst, cost] where cost is distance
        nodes: Ordered list of node IDs
        sigma: Standard deviation for Gaussian kernel
        epsilon: Threshold for sparsification

    Returns:
        NxN numpy array with exponential decay weights
    """
    N = len(nodes)
    node_to_idx = {str(n): i for i, n in enumerate(nodes)}

    adj = np.zeros((N, N), dtype=np.float32)

    for _, row in edges.iterrows():
        src_idx = node_to_idx.get(str(row["src"]))
        dst_idx = node_to_idx.get(str(row["dst"]))

        if src_idx is not None and dst_idx is not None:
            d = row["cost"]
            w = np.exp(-(d ** 2) / (sigma ** 2))

            if w >= epsilon:
                adj[src_idx, dst_idx] = w

    return adj


def thresholded_gaussian_adjacency(
    adj: np.ndarray, sigma: float = 0.1, epsilon: float = 0.5
) -> np.ndarray:
    """
    Apply Gaussian kernel and threshold to existing adjacency matrix.

    Args:
        adj: Adjacency matrix (with distances as weights)
        sigma: Standard deviation for Gaussian kernel
        epsilon: Threshold for sparsification

    Returns:
        Thresholded adjacency matrix
    """
    # Apply Gaussian kernel
    weights = np.exp(-(adj ** 2) / (sigma ** 2))

    # Threshold
    weights[weights < epsilon] = 0

    # Zero out diagonal
    np.fill_diagonal(weights, 0)

    return weights


def is_symmetric(adj: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if adjacency matrix is symmetric."""
    return np.allclose(adj, adj.T, atol=tol)


def get_degree_matrix(adj: np.ndarray) -> np.ndarray:
    """Compute degree matrix from adjacency matrix."""
    return np.diag(np.sum(adj, axis=1))


def to_sparse(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert adjacency matrix to sparse COO format.

    Returns:
        (edge_index, edge_weight) where edge_index is (2, E) and edge_weight is (E,)
    """
    rows, cols = np.nonzero(adj)
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = adj[rows, cols]
    return edge_index, edge_weight
