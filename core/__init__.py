"""
Core classes for GNN Benchmark.

This module provides the fundamental data structures and workspace management
for the GNN Benchmark toolkit.

Classes:
    DataWorkspace: Central manager for organizing and persisting datasets.
        Handles downloading, caching, and organizing data files across
        clean/, working/, and exports/ directories.

    IntermediateRepresentation: The central data object that wraps time series
        data in a unified format. Stores series DataFrame (long format),
        optional edges DataFrame for graph structure, and metadata tracking
        transform history.

    IRMetadata: Dataclass storing dataset metadata including name, frequency,
        node order, feature columns, units, and transform history. Supports
        JSON serialization.

    DatasetInfo: Static information about a dataset used by DatasetLoader
        implementations to define dataset properties.

Example:
    >>> from gnn_benchmark.core import DataWorkspace, IntermediateRepresentation
    >>> workspace = DataWorkspace("./my_workspace")
    >>> ir = workspace.load("my_dataset")
    >>> print(ir.shape)  # (T, N, C)
"""

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.core.types import IRMetadata, DatasetInfo

__all__ = [
    "DataWorkspace",
    "IntermediateRepresentation",
    "IRMetadata",
    "DatasetInfo",
]
