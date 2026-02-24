"""
Utility functions for GNN Benchmark.

This module provides common utility functions for evaluation metrics,
graph operations, and file I/O.

Evaluation metrics (all support optional boolean mask):
    mae: Mean Absolute Error
    rmse: Root Mean Squared Error
    mape: Mean Absolute Percentage Error
    mse: Mean Squared Error
    smape: Symmetric Mean Absolute Percentage Error
    r2_score: R-squared coefficient of determination

Graph utilities:
    haversine_distance: Great-circle distance between two coordinates (meters).
    adjacency_from_edges: Convert edges DataFrame to adjacency matrix.
    normalize_adjacency: Symmetric normalization D^(-1/2) A D^(-1/2).
    random_walk_matrix: Random walk normalization D^(-1) A.
    normalized_laplacian: Compute I - D^(-1/2) A D^(-1/2).
    scaled_laplacian: Scaled Laplacian for ChebNet (2L/lambda_max - I).
    exp_decay_adjacency: Exponential decay weights from distances.
    thresholded_gaussian_adjacency: Gaussian kernel with threshold.
    is_symmetric: Check if matrix is symmetric.
    get_degree_matrix: Compute degree matrix from adjacency.
    to_sparse: Convert to COO sparse format (edge_index, edge_weight).

I/O utilities:
    save_pickle, load_pickle: Pickle serialization.
    save_npz, load_npz: NumPy NPZ format.
    save_json, load_json: JSON serialization.
    ensure_dir: Create directory if it doesn't exist.
    list_files: List files matching a glob pattern.

Example:
    >>> from gnn_benchmark.utils import mae, rmse, normalize_adjacency
    >>> print(f"MAE: {mae(y_true, y_pred):.4f}")
    >>> adj_norm = normalize_adjacency(adj, add_self_loops=True)
"""

from gnn_benchmark.utils.metrics import mae, rmse, mape, mse, smape, r2_score
from gnn_benchmark.utils.graph import (
    haversine_distance,
    adjacency_from_edges,
    normalize_adjacency,
    random_walk_matrix,
    normalized_laplacian,
    scaled_laplacian,
    exp_decay_adjacency,
    thresholded_gaussian_adjacency,
    is_symmetric,
    get_degree_matrix,
    to_sparse,
)
from gnn_benchmark.utils.io import (
    save_pickle,
    load_pickle,
    save_npz,
    load_npz,
    save_json,
    load_json,
    ensure_dir,
    list_files,
)

__all__ = [
    # Metrics
    "mae",
    "rmse",
    "mape",
    "mse",
    "smape",
    "r2_score",
    # Graph utilities
    "haversine_distance",
    "adjacency_from_edges",
    "normalize_adjacency",
    "random_walk_matrix",
    "normalized_laplacian",
    "scaled_laplacian",
    "exp_decay_adjacency",
    "thresholded_gaussian_adjacency",
    "is_symmetric",
    "get_degree_matrix",
    "to_sparse",
    # I/O utilities
    "save_pickle",
    "load_pickle",
    "save_npz",
    "load_npz",
    "save_json",
    "load_json",
    "ensure_dir",
    "list_files",
]
