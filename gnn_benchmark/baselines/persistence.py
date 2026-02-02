"""Persistence (naive) baseline model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnn_benchmark.baselines.base import BaselineModel
from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class Persistence(BaselineModel):
    """
    Persistence (naive) baseline.

    Predicts the last observed value: y_hat(t+h) = y(t)

    This is a simple but strong baseline for many time series.

    Args:
        fill_method: How to handle NaN values ("forward", "zero", "mean")
    """

    fill_method: str = "forward"

    _last_values: np.ndarray | None = None
    _train_end: pd.Timestamp | None = None

    @property
    def name(self) -> str:
        return "persistence"

    def fit(self, ir: IntermediateRepresentation, train_end: pd.Timestamp) -> None:
        """Store the last values from training period."""
        self._train_end = train_end

        # Get training data
        train_data = ir.series[ir.series["ts"] <= train_end].copy()
        col = ir.feature_columns[0]

        # Pivot to wide format
        wide = train_data.pivot(index="ts", columns="node_id", values=col)
        wide = wide[ir.nodes]

        # Get last row (most recent values)
        self._last_values = wide.iloc[-1].values.copy()

        # Handle NaN based on fill method
        if self.fill_method == "forward":
            # Forward fill from earlier rows
            for i, val in enumerate(self._last_values):
                if np.isnan(val):
                    # Look back for last valid value
                    for j in range(len(wide) - 2, -1, -1):
                        if not np.isnan(wide.iloc[j, i]):
                            self._last_values[i] = wide.iloc[j, i]
                            break

        elif self.fill_method == "zero":
            self._last_values = np.nan_to_num(self._last_values, nan=0.0)

        elif self.fill_method == "mean":
            col_means = wide.mean()
            for i, val in enumerate(self._last_values):
                if np.isnan(val):
                    self._last_values[i] = col_means.iloc[i]

    def predict(
        self,
        ir: IntermediateRepresentation,
        test_start: pd.Timestamp,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """Generate persistence predictions."""
        if self._last_values is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        col = ir.feature_columns[0]
        timestamps = ir.timestamps

        try:
            test_idx = timestamps.get_loc(test_start)
        except KeyError:
            test_idx = timestamps.searchsorted(test_start)

        # Get full data for rolling predictions
        wide = ir.series.pivot(index="ts", columns="node_id", values=col)
        wide = wide[ir.nodes]
        values = wide.values

        predictions = {}

        for h in horizons:
            preds = []
            num_test = len(timestamps) - test_idx - h + 1

            for i in range(num_test):
                current_idx = test_idx + i

                # Use value at current time as prediction for t+h
                if current_idx > 0:
                    pred = values[current_idx - 1].copy()
                else:
                    pred = self._last_values.copy()

                # Handle NaN with stored last values
                nan_mask = np.isnan(pred)
                pred[nan_mask] = self._last_values[nan_mask]

                preds.append(pred)

            if preds:
                predictions[h] = np.stack(preds).flatten()

        return predictions
