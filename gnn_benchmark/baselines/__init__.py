"""Baseline models for GNN Benchmark."""

from gnn_benchmark.baselines.base import BaselineModel, BaselineResult
from gnn_benchmark.baselines.historical_average import HistoricalAverage
from gnn_benchmark.baselines.persistence import Persistence
from gnn_benchmark.baselines.sarima import SARIMA

__all__ = [
    "BaselineModel",
    "BaselineResult",
    "HistoricalAverage",
    "Persistence",
    "SARIMA",
]
