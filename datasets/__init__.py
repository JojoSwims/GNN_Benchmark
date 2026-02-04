"""
Dataset loaders for GNN Benchmark.

This module provides loaders for various spatiotemporal datasets commonly used
in GNN research. Each loader handles downloading raw data from external sources
and converting it to the intermediate representation format.

Classes:
    DatasetLoader: Abstract base class for implementing custom dataset loaders.

    BeijingAirLoader: Beijing Air Quality dataset - Beijing city (36 nodes).

    Cluster1AirLoader: Air Quality Cluster 1 - Beijing-Tianjin region (284 nodes).

    Cluster2AirLoader: Air Quality Cluster 2 - Shenzhen-Guangzhou region (163 nodes).

    PEMSBayLoader: PEMS-BAY traffic speed dataset (325 sensors, 5-min).
        Requires `gdown` package for Google Drive download.

    MetroLALoader: METR-LA traffic speed dataset (207 sensors, 5-min).
        Requires `gdown` package for Google Drive download.

    PEMS04Loader: PEMS04 traffic volume dataset (307 sensors, 5-min).
        Includes flow, occupancy, and speed features.

    PEMS08Loader: PEMS08 traffic volume dataset (170 sensors, 5-min).
        Includes flow, occupancy, and speed features.

    ElergoneLoader: Electricity Load Diagrams dataset (370 clients, 15-min).
        No graph structure provided.

Example:
    >>> from gnn_benchmark import DataWorkspace
    >>> from gnn_benchmark.datasets import BeijingAirLoader, Cluster1AirLoader
    >>> workspace = DataWorkspace("./my_workspace")
    >>> # Load multiple air quality datasets simultaneously
    >>> beijing_ir = BeijingAirLoader().prepare(workspace)
    >>> cluster1_ir = Cluster1AirLoader().prepare(workspace)
"""

from gnn_benchmark.datasets.base import DatasetLoader
from gnn_benchmark.datasets.beijing_air import (
    BeijingAirLoader,
    Cluster1AirLoader,
    Cluster2AirLoader,
)
from gnn_benchmark.datasets.elergone import ElergoneLoader
from gnn_benchmark.datasets.pems_speed import PEMSBayLoader, MetroLALoader
from gnn_benchmark.datasets.pems_volume import PEMS04Loader, PEMS08Loader

__all__ = [
    "DatasetLoader",
    "BeijingAirLoader",
    "Cluster1AirLoader",
    "Cluster2AirLoader",
    "ElergoneLoader",
    "PEMSBayLoader",
    "MetroLALoader",
    "PEMS04Loader",
    "PEMS08Loader",
]
