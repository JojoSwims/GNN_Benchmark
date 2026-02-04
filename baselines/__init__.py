"""
Baseline models for GNN Benchmark.

This module provides simple baseline forecasting models for comparison with
more sophisticated GNN approaches. All baselines follow a common interface
with fit(), predict(), and evaluate() methods.

Classes:
    BaselineModel: Abstract base class for implementing custom baselines.
    BaselineResult: Dataclass containing predictions, actuals, and metrics.

    Model implementations:
        Persistence: Naive baseline that predicts the last observed value.
            y_hat(t+h) = y(t). Simple but strong baseline for many series.

        HistoricalAverage: Predicts using the average from the same period
            in historical data. Configurable period (default: weekly).

        SARIMA: Seasonal ARIMA model fitted independently per node.
            Optional Kalman imputation for missing values. Requires statsmodels.

Example:
    >>> from gnn_benchmark.baselines import Persistence
    >>> import pandas as pd
    >>> baseline = Persistence(fill_method="forward")
    >>> result = baseline.evaluate(
    ...     ir,
    ...     train_end=pd.Timestamp("2015-01-01"),
    ...     val_end=pd.Timestamp("2015-03-01"),
    ...     horizons=[1, 3, 6, 12]
    ... )
    >>> print(f"MAE: {result.metrics['mae']:.4f}")
"""

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
