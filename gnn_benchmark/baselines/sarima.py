"""SARIMA baseline model with optional Kalman imputation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnn_benchmark.baselines.base import BaselineModel
from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class SARIMA(BaselineModel):
    """
    SARIMA baseline with Kalman filter imputation.

    Fits independent SARIMA models for each node/sensor.

    Args:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        impute_missing: Whether to impute missing values before fitting
        minutes_per_step: Data frequency in minutes (for computing seasonal periods)
    """

    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 288)  # Default: daily
    impute_missing: bool = True
    minutes_per_step: int = 5

    _models: dict = field(default_factory=dict, init=False, repr=False)
    _train_end: pd.Timestamp | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"sarima_{self.order}_{self.seasonal_order}"

    def fit(self, ir: IntermediateRepresentation, train_end: pd.Timestamp) -> None:
        """Fit SARIMA models for each node."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError(
                "SARIMA requires statsmodels. Install with: pip install statsmodels"
            )

        self._train_end = train_end
        self._models = {}

        # Get training data
        train_data = ir.series[ir.series["ts"] <= train_end].copy()
        col = ir.feature_columns[0]

        # Pivot to wide format
        wide = train_data.pivot(index="ts", columns="node_id", values=col)
        wide = wide[ir.nodes]

        # Optionally impute missing values
        if self.impute_missing:
            wide = self._kalman_impute(wide)

        # Fit model for each node
        print(f"Fitting SARIMA models for {len(ir.nodes)} nodes...")
        for i, node in enumerate(ir.nodes):
            series = wide[node].values

            try:
                model = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._models[node] = model.fit(disp=False)
            except Exception as e:
                print(f"Warning: Failed to fit SARIMA for node {node}: {e}")
                self._models[node] = None

            if (i + 1) % 50 == 0:
                print(f"  Fitted {i + 1}/{len(ir.nodes)} models")

        print("SARIMA fitting complete.")

    def predict(
        self,
        ir: IntermediateRepresentation,
        test_start: pd.Timestamp,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """Generate SARIMA predictions."""
        if not self._models:
            raise RuntimeError("Model not fitted. Call fit() first.")

        col = ir.feature_columns[0]
        timestamps = ir.timestamps

        try:
            test_idx = timestamps.get_loc(test_start)
        except KeyError:
            test_idx = timestamps.searchsorted(test_start)

        max_horizon = max(horizons)
        predictions = {h: [] for h in horizons}

        # Get full data for forecasting
        wide = ir.series.pivot(index="ts", columns="node_id", values=col)
        wide = wide[ir.nodes]

        # For each test point, forecast
        num_test = len(timestamps) - test_idx - max_horizon + 1

        for i in range(num_test):
            current_idx = test_idx + i

            for node in ir.nodes:
                model_result = self._models.get(node)

                if model_result is not None:
                    try:
                        # Forecast from current position
                        forecast = model_result.get_forecast(steps=max_horizon)
                        pred_values = forecast.predicted_mean

                        for h in horizons:
                            predictions[h].append(pred_values[h - 1])
                    except Exception:
                        # Fall back to last value
                        last_val = wide.iloc[current_idx - 1][node]
                        for h in horizons:
                            predictions[h].append(last_val)
                else:
                    # Use persistence for failed models
                    last_val = wide.iloc[current_idx - 1][node] if current_idx > 0 else 0
                    for h in horizons:
                        predictions[h].append(last_val)

        # Convert to arrays
        for h in horizons:
            predictions[h] = np.array(predictions[h])

        return predictions

    def _kalman_impute(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using Kalman filter."""
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            # Fall back to linear interpolation
            return wide.interpolate(method="linear", limit_direction="both")

        result = wide.copy()

        steps_per_day = 24 * 60 // self.minutes_per_step

        for col in wide.columns:
            series = wide[col].values

            if np.isnan(series).any():
                try:
                    model = UnobservedComponents(
                        series,
                        level="local level",
                        seasonal=steps_per_day,
                        stochastic_seasonal=True,
                    )
                    fitted = model.fit(disp=False)
                    result[col] = fitted.smoothed_state[0]
                except Exception:
                    # Fall back to interpolation
                    result[col] = wide[col].interpolate(
                        method="linear", limit_direction="both"
                    )

        return result
