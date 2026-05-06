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

    EULoadLoader: ENTSO-E European zonal electricity load (hourly, MW).
        Undirected, unweighted cross-zone interconnection edges (static)
        and/or hourly directed cross-zone flow snapshots (dynamic),
        selectable via ``edges_mode``.
        Requires `gdown` package for Google Drive download.

    NYCovidLoader: NYT COVID-19 county-level dataset (2000+ counties, daily).
        Haversine distance graph.

    LamaHCELoader: LamaH-CE Central Europe streamflow dataset (up to 859 gauges).
        Downloads the Google Drive mirror with gdown when data_root is omitted.
        River network topology edges and qobs + 8 ERA5-Land forcings (C=9).

    LamaHCEDynamicLoader: Deprecated alias for LamaHCELoader.

    NOAABuoyLoader: NOAA NDBC ocean buoy network, North Atlantic & Gulf of
        Mexico, 2020-2023 (hourly). 4 features: WTMP, WSPD, WVHT, PRES.
        Haversine distance graph.

    DivvyBikeshareLoader: Divvy Chicago bikeshare, March 2021 (677 stations,
        hourly). Node features: departures, arrivals. Static
        fully-connected haversine graph. Hourly directed trip-count dynamic
        graph bucketed by trip end-time (selectable via ``edges_mode``).
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
from gnn_benchmark.datasets.eu_load import EULoadLoader
from gnn_benchmark.datasets.ny_covid import NYCovidLoader
from gnn_benchmark.datasets.pems_speed import PEMSBayLoader, MetroLALoader
from gnn_benchmark.datasets.lamah_ce import LamaHCELoader, LamaHCEDynamicLoader
from gnn_benchmark.datasets.pems_volume import PEMS04Loader, PEMS08Loader
from gnn_benchmark.datasets.noaa_buoy import NOAABuoyLoader
from gnn_benchmark.datasets.divvy_bikeshare import DivvyBikeshareLoader


__all__ = [
    "DatasetLoader",
    "BeijingAirLoader",
    "Cluster1AirLoader",
    "Cluster2AirLoader",
    "ElergoneLoader",
    "EULoadLoader",
    "NYCovidLoader",
    "PEMSBayLoader",
    "MetroLALoader",
    "PEMS04Loader",
    "PEMS08Loader",
    "NOAABuoyLoader",
    "LamaHCELoader",
    "LamaHCEDynamicLoader",
    "DivvyBikeshareLoader",
]
