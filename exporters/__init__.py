"""
Model-specific exporters for GNN Benchmark.

This module provides exporters that convert the IntermediateRepresentation
to formats expected by various GNN models. Each exporter handles windowing,
train/val/test splitting, and format-specific file generation.

Classes:
    ModelExporter: Abstract base class for implementing custom exporters.

    Configuration classes:
        WindowConfig: Sliding window parameters (input_length, horizon, y_start).
        SplitConfig: Train/val/test split ratios.
        ExportResult: Result containing file paths and metadata.

    Exporter implementations:
        STAEformerExporter: Full tensor + index arrays for runtime windowing.
        GraphWaveNetExporter: Pre-windowed train/val/test splits with adjacency.
        ASTGCNExporter: Full tensor + windowed splits + sensor IDs.
        MTGNNExporter: HDF5 format with adjacency and windowed splits.
        GTSExporter: Format for GTS model.
        D2STGCNExporter: Format for D2STGCN model.

Example:
    >>> from gnn_benchmark.exporters import STAEformerExporter, WindowConfig
    >>> exporter = STAEformerExporter()
    >>> result = exporter.export_from_workspace(
    ...     ir,
    ...     window_config=WindowConfig(input_length=12, horizon=12)
    ... )
    >>> print(result.files)
"""

from gnn_benchmark.exporters.base import ModelExporter, WindowConfig, SplitConfig, ExportResult
from gnn_benchmark.exporters.staeformer import STAEformerExporter
from gnn_benchmark.exporters.graph_wavenet import GraphWaveNetExporter
from gnn_benchmark.exporters.astgcn import ASTGCNExporter
from gnn_benchmark.exporters.mtgnn import MTGNNExporter
from gnn_benchmark.exporters.gts import GTSExporter
from gnn_benchmark.exporters.d2stgcn import D2STGCNExporter

__all__ = [
    "ModelExporter",
    "WindowConfig",
    "SplitConfig",
    "ExportResult",
    "STAEformerExporter",
    "GraphWaveNetExporter",
    "ASTGCNExporter",
    "MTGNNExporter",
    "GTSExporter",
    "D2STGCNExporter",
]
