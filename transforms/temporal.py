"""Temporal transforms for adding time features and filtering."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from gnn_benchmark.transforms.base import Transform

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class AddTimeFeatures(Transform):
    """
    Add temporal feature columns to the series.

    Features:
        - time_of_day: Normalized time of day [0, 1)
        - day_of_week: Day of week 0-6 (Monday=0)
        - time_of_year: Normalized day of year [0, 1)
    """

    time_of_day: bool = True
    day_of_week: bool = True
    time_of_year: bool = False

    @property
    def description(self) -> str:
        features = []
        if self.time_of_day:
            features.append("tod")
        if self.day_of_week:
            features.append("dow")
        if self.time_of_year:
            features.append("toy")
        return f"AddTimeFeatures({', '.join(features)})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        ts = pd.to_datetime(ir.series["ts"])

        if self.time_of_day:
            # Normalized time of day: seconds since midnight / seconds in day
            seconds_in_day = 24 * 60 * 60
            seconds_since_midnight = (
                ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
            )
            ir.series["time_of_day"] = seconds_since_midnight / seconds_in_day
            if "time_of_day" not in ir.metadata.feature_columns:
                ir.metadata.feature_columns.append("time_of_day")

        if self.day_of_week:
            # Day of week: 0=Monday, 6=Sunday
            ir.series["day_of_week"] = ts.dt.dayofweek
            if "day_of_week" not in ir.metadata.feature_columns:
                ir.metadata.feature_columns.append("day_of_week")

        if self.time_of_year:
            # Normalized day of year: day / days in year
            day_of_year = ts.dt.dayofyear
            days_in_year = ts.dt.is_leap_year.map({True: 366, False: 365})
            ir.series["time_of_year"] = day_of_year / days_in_year
            if "time_of_year" not in ir.metadata.feature_columns:
                ir.metadata.feature_columns.append("time_of_year")


@dataclass
class FilterHours(Transform):
    """
    Keep only specific hours of day.

    Useful for reducing data to specific time points (e.g., midnight and noon).
    """

    hours: list[int]  # e.g., [0, 12] for midnight and noon

    @property
    def description(self) -> str:
        return f"FilterHours({self.hours})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        ts = pd.to_datetime(ir.series["ts"])
        mask = ts.dt.hour.isin(self.hours)
        ir.series = ir.series[mask].reset_index(drop=True)

        # Update mask if present
        if ir.mask is not None:
            mask_ts = pd.to_datetime(ir.mask["ts"])
            mask_filter = mask_ts.dt.hour.isin(self.hours)
            ir.mask = ir.mask[mask_filter].reset_index(drop=True)


@dataclass
class FilterDateRange(Transform):
    """
    Keep only timestamps within a date range.

    Both start and end are inclusive.
    """

    start: pd.Timestamp | str | None = None
    end: pd.Timestamp | str | None = None

    @property
    def description(self) -> str:
        start_str = str(self.start) if self.start else "None"
        end_str = str(self.end) if self.end else "None"
        return f"FilterDateRange(start={start_str}, end={end_str})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        ts = pd.to_datetime(ir.series["ts"])

        mask = pd.Series([True] * len(ts))

        if self.start is not None:
            start = pd.Timestamp(self.start)
            mask = mask & (ts >= start)

        if self.end is not None:
            end = pd.Timestamp(self.end)
            mask = mask & (ts <= end)

        ir.series = ir.series[mask].reset_index(drop=True)

        # Update mask if present
        if ir.mask is not None:
            mask_ts = pd.to_datetime(ir.mask["ts"])
            mask_filter = pd.Series([True] * len(mask_ts))

            if self.start is not None:
                mask_filter = mask_filter & (mask_ts >= pd.Timestamp(self.start))
            if self.end is not None:
                mask_filter = mask_filter & (mask_ts <= pd.Timestamp(self.end))

            ir.mask = ir.mask[mask_filter].reset_index(drop=True)


@dataclass
class Resample(Transform):
    """
    Resample time series to a different frequency.

    Aggregates values using the specified method (mean, sum, etc.).
    """

    frequency: str  # e.g., "1H", "30T", "1D"
    method: str = "mean"  # Aggregation method: mean, sum, min, max, first, last

    @property
    def description(self) -> str:
        return f"Resample(freq={self.frequency}, method={self.method})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        feature_cols = ir.feature_columns

        # Set timestamp as index for resampling
        ir.series["ts"] = pd.to_datetime(ir.series["ts"])

        # Group by node_id and resample
        resampled_dfs = []

        for node_id in ir.series["node_id"].unique():
            node_data = ir.series[ir.series["node_id"] == node_id].copy()
            node_data = node_data.set_index("ts")

            # Resample feature columns
            if self.method == "mean":
                resampled = node_data[feature_cols].resample(self.frequency).mean()
            elif self.method == "sum":
                resampled = node_data[feature_cols].resample(self.frequency).sum()
            elif self.method == "min":
                resampled = node_data[feature_cols].resample(self.frequency).min()
            elif self.method == "max":
                resampled = node_data[feature_cols].resample(self.frequency).max()
            elif self.method == "first":
                resampled = node_data[feature_cols].resample(self.frequency).first()
            elif self.method == "last":
                resampled = node_data[feature_cols].resample(self.frequency).last()
            else:
                raise ValueError(f"Unknown resample method: {self.method}")

            resampled = resampled.reset_index()
            resampled["node_id"] = node_id
            resampled_dfs.append(resampled)

        ir.series = pd.concat(resampled_dfs, ignore_index=True)
        ir.series = ir.series[["ts", "node_id"] + feature_cols]
        ir.series = ir.series.sort_values(["ts", "node_id"]).reset_index(drop=True)

        # Update metadata frequency
        ir.metadata.frequency = self.frequency

        # Invalidate mask (would need to be recomputed)
        ir.mask = None
