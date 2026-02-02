"""Imputation transforms for handling missing values."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from gnn_benchmark.transforms.base import Transform

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class FillZeros(Transform):
    """
    Replace NaN values with zeros and generate observation mask.

    The mask DataFrame tracks which values were originally observed (True)
    vs imputed (False).
    """

    columns: list[str] | None = None  # None = all feature columns

    @property
    def description(self) -> str:
        cols = self.columns or "all"
        return f"FillZeros(columns={cols})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        cols = self.columns or ir.feature_columns

        # Create or update mask
        if ir.mask is None:
            # Create mask from current NaN positions
            mask_records = []
            for _, row in ir.series.iterrows():
                is_observed = all(pd.notna(row[c]) for c in cols)
                mask_records.append(
                    {
                        "ts": row["ts"],
                        "node_id": row["node_id"],
                        "is_observed": is_observed,
                    }
                )
            ir.mask = pd.DataFrame(mask_records)
        else:
            # Update existing mask
            for col in cols:
                ir.mask["is_observed"] = ir.mask["is_observed"] & ir.series[col].notna()

        # Fill NaN with zeros
        for col in cols:
            ir.series[col] = ir.series[col].fillna(0.0)


@dataclass
class ForwardFill(Transform):
    """
    Forward fill NaN values within each node's time series.

    Fills missing values with the last observed value for each node.
    """

    columns: list[str] | None = None  # None = all feature columns
    limit: int | None = None  # Maximum number of consecutive NaNs to fill

    @property
    def description(self) -> str:
        cols = self.columns or "all"
        limit_str = f", limit={self.limit}" if self.limit else ""
        return f"ForwardFill(columns={cols}{limit_str})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        cols = self.columns or ir.feature_columns

        # Create mask before filling
        if ir.mask is None:
            mask_records = []
            for _, row in ir.series.iterrows():
                is_observed = all(pd.notna(row[c]) for c in cols)
                mask_records.append(
                    {
                        "ts": row["ts"],
                        "node_id": row["node_id"],
                        "is_observed": is_observed,
                    }
                )
            ir.mask = pd.DataFrame(mask_records)

        # Sort by node_id and timestamp for proper forward fill
        ir.series = ir.series.sort_values(["node_id", "ts"])

        # Forward fill within each node group
        for col in cols:
            ir.series[col] = ir.series.groupby("node_id")[col].ffill(limit=self.limit)

        # Restore original order (by timestamp, then node_id)
        ir.series = ir.series.sort_values(["ts", "node_id"]).reset_index(drop=True)


@dataclass
class KalmanImpute(Transform):
    """
    Impute missing values using Kalman filter with seasonal components.

    Uses statsmodels UnobservedComponents with daily and weekly seasonality.
    This is computationally expensive for large datasets.
    """

    columns: list[str] | None = None
    minutes_per_step: int = 5  # Frequency of data

    # Computed after fitting
    _fitted: bool = field(default=False, init=False, repr=False)

    @property
    def description(self) -> str:
        cols = self.columns or "all"
        return f"KalmanImpute(columns={cols}, freq={self.minutes_per_step}min)"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            raise ImportError(
                "KalmanImpute requires statsmodels. "
                "Install with: pip install statsmodels"
            )

        cols = self.columns or ir.feature_columns

        # Create mask before imputation
        if ir.mask is None:
            mask_records = []
            for _, row in ir.series.iterrows():
                is_observed = all(pd.notna(row[c]) for c in cols)
                mask_records.append(
                    {
                        "ts": row["ts"],
                        "node_id": row["node_id"],
                        "is_observed": is_observed,
                    }
                )
            ir.mask = pd.DataFrame(mask_records)

        # Calculate seasonal periods
        steps_per_day = 24 * 60 // self.minutes_per_step
        steps_per_week = 7 * steps_per_day

        # Process each node separately
        nodes = ir.series["node_id"].unique()

        for node in nodes:
            node_mask = ir.series["node_id"] == node
            node_data = ir.series.loc[node_mask].sort_values("ts")

            for col in cols:
                series = node_data[col].values
                if pd.isna(series).any():
                    try:
                        # Fit Kalman filter with seasonal components
                        model = UnobservedComponents(
                            series,
                            level="local level",
                            seasonal=steps_per_day,
                            stochastic_seasonal=True,
                        )
                        result = model.fit(disp=False)
                        smoothed = result.smoothed_state[0]

                        # Update values in the original series
                        ir.series.loc[node_mask, col] = smoothed
                    except Exception:
                        # Fall back to linear interpolation if Kalman fails
                        ir.series.loc[node_mask, col] = (
                            ir.series.loc[node_mask, col].interpolate(method="linear")
                        )

        self._fitted = True
