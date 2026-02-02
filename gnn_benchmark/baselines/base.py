"""Base BaselineModel class and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class BaselineResult:
    """Result from baseline model evaluation."""

    predictions: np.ndarray  # Shape depends on model
    actuals: np.ndarray
    metrics: dict[str, float]  # Overall metrics: MAE, RMSE, MAPE
    per_horizon_metrics: dict[int, dict[str, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaselineModel(ABC):
    """
    Base class for baseline forecasting models.

    Provides a common interface for fitting and evaluating simple
    forecasting baselines.

    Subclasses must implement:
        - name: Property returning the model name
        - fit: Method to fit the model on training data
        - predict: Method to generate predictions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""

    @abstractmethod
    def fit(
        self, ir: "IntermediateRepresentation", train_end: pd.Timestamp
    ) -> None:
        """
        Fit the model on training data.

        Args:
            ir: IntermediateRepresentation with the data
            train_end: End timestamp for training data (inclusive)
        """

    @abstractmethod
    def predict(
        self,
        ir: "IntermediateRepresentation",
        test_start: pd.Timestamp,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Generate predictions for the test period.

        Args:
            ir: IntermediateRepresentation with the data
            test_start: Start timestamp for test period
            horizons: List of forecast horizons to predict

        Returns:
            Dictionary mapping horizon -> predictions array
        """

    def evaluate(
        self,
        ir: "IntermediateRepresentation",
        train_end: pd.Timestamp,
        val_end: pd.Timestamp,
        horizons: list[int],
        target_column: str | None = None,
    ) -> BaselineResult:
        """
        Full fit-predict-evaluate pipeline.

        Args:
            ir: IntermediateRepresentation with the data
            train_end: End timestamp for training data
            val_end: End timestamp for validation data (test starts after)
            horizons: List of forecast horizons
            target_column: Column to predict (default: first feature column)

        Returns:
            BaselineResult with predictions, actuals, and metrics
        """
        from gnn_benchmark.utils.metrics import mae, mape, rmse

        # Fit on training data
        self.fit(ir, train_end)

        # Get test start
        test_start = val_end + pd.Timedelta(ir.metadata.frequency)

        # Get predictions
        predictions = self.predict(ir, test_start, horizons)

        # Get actuals for comparison
        col = target_column or ir.feature_columns[0]
        test_data = ir.series[ir.series["ts"] > val_end].copy()

        # Compute metrics per horizon
        per_horizon_metrics = {}
        all_preds = []
        all_actuals = []

        for h in horizons:
            if h in predictions:
                pred = predictions[h]
                # Get actual values at horizon h
                actual = self._get_actuals_at_horizon(ir, test_start, h, col)

                if actual is not None and len(pred) > 0:
                    # Align lengths
                    min_len = min(len(pred), len(actual))
                    pred = pred[:min_len]
                    actual = actual[:min_len]

                    # Compute metrics
                    mask = ~np.isnan(actual) & ~np.isnan(pred)
                    if mask.sum() > 0:
                        per_horizon_metrics[h] = {
                            "mae": mae(actual[mask], pred[mask]),
                            "rmse": rmse(actual[mask], pred[mask]),
                            "mape": mape(actual[mask], pred[mask]),
                        }

                        all_preds.extend(pred[mask])
                        all_actuals.extend(actual[mask])

        # Compute overall metrics
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        overall_metrics = {}
        if len(all_preds) > 0:
            overall_metrics = {
                "mae": mae(all_actuals, all_preds),
                "rmse": rmse(all_actuals, all_preds),
                "mape": mape(all_actuals, all_preds),
            }

        return BaselineResult(
            predictions=all_preds,
            actuals=all_actuals,
            metrics=overall_metrics,
            per_horizon_metrics=per_horizon_metrics,
            metadata={"horizons": horizons, "target_column": col},
        )

    def _get_actuals_at_horizon(
        self,
        ir: "IntermediateRepresentation",
        test_start: pd.Timestamp,
        horizon: int,
        column: str,
    ) -> np.ndarray | None:
        """Get actual values at a specific horizon from test start."""
        timestamps = ir.timestamps
        test_idx = timestamps.get_loc(test_start) if test_start in timestamps else None

        if test_idx is None:
            return None

        # Get timestamps at horizon
        target_idx = test_idx + horizon - 1
        if target_idx >= len(timestamps):
            return None

        target_ts = timestamps[target_idx:]
        test_data = ir.series[ir.series["ts"].isin(target_ts)]

        # Pivot to wide format
        wide = test_data.pivot(index="ts", columns="node_id", values=column)
        wide = wide[ir.nodes]

        return wide.values.flatten()
