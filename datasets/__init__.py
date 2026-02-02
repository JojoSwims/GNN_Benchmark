"""Dataset loaders for GNN Benchmark."""

from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.datasets.beijing_air import BeijingAirLoader
from gnn_benchmark.datasets.elergone import ElergoneLoader
from gnn_benchmark.datasets.pems_speed import PEMSBayLoader, MetroLALoader
from gnn_benchmark.datasets.pems_volume import PEMS04Loader, PEMS08Loader

__all__ = [
    "DatasetLoader",
    "BeijingAirLoader",
    "ElergoneLoader",
    "PEMSBayLoader",
    "MetroLALoader",
    "PEMS04Loader",
    "PEMS08Loader",
]
