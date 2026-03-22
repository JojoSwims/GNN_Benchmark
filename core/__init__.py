"""Core classes for GNN Benchmark."""

from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.core.types import IRMetadata, DatasetInfo, WindowConfig, SplitConfig

__all__ = [
    "DataWorkspace",
    "IntermediateRepresentation",
    "IRMetadata",
    "DatasetInfo",
    "WindowConfig",
    "SplitConfig",
]
