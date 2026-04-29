"""
GNN Benchmark - A toolkit for benchmarking Graph Neural Networks on time series data.

This package provides a clean interface for:
1. Loading datasets from external sources
2. Preparing sliding-window (x, y) train/val/test tensors
3. Defining a model contract (fit / predict)
4. Evaluating model predictions against ground truth

Example usage:

    from gnn_benchmark import BenchmarkRunner, LastValueModel

    runner = BenchmarkRunner(
        workspace_dir="./benchmark_workspace",
        datasets=["metr-la"],
    )
    result = runner.run(LastValueModel())
    print(result.summary())
"""

__version__ = "0.1.0"

# Runner
from gnn_benchmark.benchmark import BenchmarkRunner

# Core classes
from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.core.types import IRMetadata, DatasetInfo, WindowConfig, SplitConfig

# Model contract
from gnn_benchmark.models import BenchmarkModel, TrainingHistory, LastValueModel

# Re-export submodules for convenient access
from gnn_benchmark import datasets
from gnn_benchmark import utils
from gnn_benchmark import models

__all__ = [
    # Version
    "__version__",
    # Runner
    "BenchmarkRunner",
    # Core classes
    "DataWorkspace",
    "IntermediateRepresentation",
    "IRMetadata",
    "DatasetInfo",
    "WindowConfig",
    "SplitConfig",
    # Model contract
    "BenchmarkModel",
    "TrainingHistory",
    "LastValueModel",
    # Submodules
    "datasets",
    "utils",
    "models",
]
