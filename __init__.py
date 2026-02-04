"""
GNN Benchmark - A toolkit for benchmarking Graph Neural Networks on time series data.

This package provides a clean, modular interface for:
1. Loading datasets from external sources
2. Applying transforms to intermediate representations
3. Exporting data in model-specific formats
4. Running baseline models for comparison

Example usage:

    from gnn_benchmark import DataWorkspace
    from gnn_benchmark.datasets import BeijingAirLoader
    from gnn_benchmark.transforms import FillZeros, AddTimeFeatures
    from gnn_benchmark.exporters import STAEformerExporter, WindowConfig

    # Set up workspace
    workspace = DataWorkspace("./my_workspace")

    # Prepare dataset
    loader = BeijingAirLoader(subdivision="beijing")
    ir = loader.prepare(workspace)

    # Apply transforms
    ir.apply(FillZeros())
    ir.apply(AddTimeFeatures(time_of_day=True, day_of_week=True))

    # Export for specific model
    exporter = STAEformerExporter()
    result = exporter.export(
        ir,
        window_config=WindowConfig(input_length=12, horizon=12)
    )
    print(f"Exported to: {result.files}")
"""

__version__ = "0.1.0"

# Core classes
from gnn_benchmark.core.workspace import DataWorkspace
from gnn_benchmark.core.intermediate import IntermediateRepresentation
from gnn_benchmark.core.types import IRMetadata, DatasetInfo

# Re-export submodules for convenient access
from gnn_benchmark import datasets
from gnn_benchmark import transforms
from gnn_benchmark import exporters
from gnn_benchmark import baselines
from gnn_benchmark import utils

__all__ = [
    # Version
    "__version__",
    # Core classes
    "DataWorkspace",
    "IntermediateRepresentation",
    "IRMetadata",
    "DatasetInfo",
    # Submodules
    "datasets",
    "transforms",
    "exporters",
    "baselines",
    "utils",
]
