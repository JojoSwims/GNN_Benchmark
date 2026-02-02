"""Historical Average baseline model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnn_benchmark.baselines.base import BaselineModel
from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class HistoricalAverage(BaselineModel):
    """
    Historical Average baseline.

    Predicts using the average of values from the same time period
    (e.g., same hour of week) from historical data.

    Args:
        period: Number of timesteps in one period (default: 168 for 1 week at hourly)
    """

    period: int = 168  # Default: 1 week at hourly frequency

    _history: np.ndarray | None = None
    _train_end: pd.Timestamp | None = None

    @property
    def name(self) -> str:
        return f"historical_average_{self.period}"

    def fit(self, ir: IntermediateRepresentation, train_end: pd.Timestamp) -> None:
        """Compute historical averages from training data."""
        self._train_end = train_end

        # Get training data
        train_data = ir.series[ir.series["ts"] <= train_end].copy()
        col = ir.feature_columns[0]

        # Pivot to wide format (T, N)
        wide = train_data.pivot(index="ts", columns="node_id", values=col)
        wide = wide[ir.nodes]

        values = wide.values
        T, N = values.shape

        # Compute average for each position in the period
        self._history = np.zeros((self.period, N))
        counts = np.zeros((self.period, N))

        for t in range(T):
            period_idx = t % self.period
            mask = ~np.isnan(values[t])
            self._history[period_idx, mask] += values[t, mask]
            counts[period_idx, mask] += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        self._history /= counts

    def predict(
        self,
        ir: IntermediateRepresentation,
        test_start: pd.Timestamp,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """Generate predictions using historical averages."""
        if self._history is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        timestamps = ir.timestamps
        try:
            test_idx = timestamps.get_loc(test_start)
        except KeyError:
            # Find nearest timestamp
            test_idx = timestamps.searchsorted(test_start)

        predictions = {}

        for h in horizons:
            # Predict for all test timestamps at horizon h
            preds = []
            num_test = len(timestamps) - test_idx - h + 1

            for i in range(num_test):
                current_idx = test_idx + i
                target_idx = current_idx + h - 1

                if target_idx >= len(timestamps):
                    break

                # Get period index for target
                period_idx = target_idx % self.period
                preds.append(self._history[period_idx])

            if preds:
                predictions[h] = np.stack(preds).flatten()

        return predictions
