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

    NYISOLoader: NYISO Integrated Load dataset (11 regions, hourly).
        No graph structure provided.

    NYCovidLoader: NYT COVID-19 county-level dataset (2000+ counties, daily).
        Haversine distance graph.

    MelPedsLoader: Melbourne Pedestrian Counts dataset (55 sensors, hourly).
        Haversine distance graph.

    LamaHCELoader: LamaH-CE Central Europe streamflow dataset (up to 859 gauges).
        Requires local copy of Zenodo data. River network topology edges.
        Includes dynamic met + static catchment features (C=30).

    LamaHCEDynamicLoader: LamaH-CE dynamic-only variant (up to 859 gauges).
        Same river network, but only qobs + 8 ERA5-Land met forcings (C=9).

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
from gnn_benchmark.datasets.iso import NYISOLoader
from gnn_benchmark.datasets.melpeds import MelPedsLoader
from gnn_benchmark.datasets.ny_covid import NYCovidLoader
from gnn_benchmark.datasets.pems_speed import PEMSBayLoader, MetroLALoader
from gnn_benchmark.datasets.lamah_ce import LamaHCELoader
from gnn_benchmark.datasets.lamah_ce_dynamic import LamaHCEDynamicLoader
from gnn_benchmark.datasets.pems_volume import PEMS04Loader, PEMS08Loader

__all__ = [
    "DatasetLoader",
    "BeijingAirLoader",
    "Cluster1AirLoader",
    "Cluster2AirLoader",
    "ElergoneLoader",
    "NYISOLoader",
    "NYCovidLoader",
    "MelPedsLoader",
    "PEMSBayLoader",
    "MetroLALoader",
    "PEMS04Loader",
    "PEMS08Loader",
    "LamaHCELoader",
    "LamaHCEDynamicLoader",
]
