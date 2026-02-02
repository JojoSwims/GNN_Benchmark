"""Utility functions for GNN Benchmark."""

from gnn_benchmark.utils.metrics import mae, rmse, mape, mse, smape, r2_score
from gnn_benchmark.utils.graph import (
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
