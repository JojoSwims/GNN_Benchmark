"""Normalization transforms for feature scaling."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from gnn_benchmark.transforms.base import Transform

if TYPE_CHECKING:
    from gnn_benchmark.core.intermediate import IntermediateRepresentation


@dataclass
class ZScoreNormalize(Transform):
    """
    Z-score normalization using training statistics.

    Computes mean and std from training data (ts <= train_end) and
    applies normalization to the entire dataset.

    The computed statistics are stored in means_ and stds_ after fitting.
    """

    train_end: pd.Timestamp | str
    columns: list[str] | None = None  # None = all feature columns
    eps: float = 1e-8  # Small value to prevent division by zero

    # Stored after fitting
    means_: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    stds_: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def description(self) -> str:
        train_end = pd.Timestamp(self.train_end)
        cols = self.columns or "all"
        return f"ZScoreNormalize(train_end={train_end.isoformat()}, columns={cols})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        cols = self.columns or ir.feature_columns
        train_end = pd.Timestamp(self.train_end)

        # Get training data
        ts = pd.to_datetime(ir.series["ts"])
        train_mask = ts <= train_end

        # Compute stats from training data
        for col in cols:
            train_values = ir.series.loc[train_mask, col]
            self.means_[col] = float(train_values.mean())
            self.stds_[col] = float(train_values.std())

            # Apply normalization to all data
            mean = self.means_[col]
            std = max(self.stds_[col], self.eps)
            ir.series[col] = (ir.series[col] - mean) / std

        # Store stats in metadata for later denormalization
        ir.metadata.extra["zscore_means"] = self.means_
        ir.metadata.extra["zscore_stds"] = self.stds_


@dataclass
class MinMaxNormalize(Transform):
    """
    Min-max normalization to [0, 1] range using training statistics.

    Computes min and max from training data (ts <= train_end) and
    applies normalization to the entire dataset.
    """

    train_end: pd.Timestamp | str
    columns: list[str] | None = None  # None = all feature columns
    eps: float = 1e-8  # Small value to prevent division by zero

    # Stored after fitting
    mins_: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    maxs_: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def description(self) -> str:
        train_end = pd.Timestamp(self.train_end)
        cols = self.columns or "all"
        return f"MinMaxNormalize(train_end={train_end.isoformat()}, columns={cols})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        cols = self.columns or ir.feature_columns
        train_end = pd.Timestamp(self.train_end)

        # Get training data
        ts = pd.to_datetime(ir.series["ts"])
        train_mask = ts <= train_end

        # Compute stats from training data
        for col in cols:
            train_values = ir.series.loc[train_mask, col]
            self.mins_[col] = float(train_values.min())
            self.maxs_[col] = float(train_values.max())

            # Apply normalization to all data
            min_val = self.mins_[col]
            max_val = self.maxs_[col]
            range_val = max(max_val - min_val, self.eps)
            ir.series[col] = (ir.series[col] - min_val) / range_val

        # Store stats in metadata for later denormalization
        ir.metadata.extra["minmax_mins"] = self.mins_
        ir.metadata.extra["minmax_maxs"] = self.maxs_


@dataclass
class Denormalize(Transform):
    """
    Reverse normalization using stored statistics.

    Reads statistics from IR metadata and applies inverse transformation.
    Works with either ZScore or MinMax normalization.
    """

    method: str = "zscore"  # "zscore" or "minmax"
    columns: list[str] | None = None

    @property
    def description(self) -> str:
        cols = self.columns or "all"
        return f"Denormalize(method={self.method}, columns={cols})"

    def __call__(self, ir: "IntermediateRepresentation") -> None:
        cols = self.columns or ir.feature_columns

        if self.method == "zscore":
            if "zscore_means" not in ir.metadata.extra:
                raise ValueError("No ZScore normalization stats found in metadata")

            means = ir.metadata.extra["zscore_means"]
            stds = ir.metadata.extra["zscore_stds"]

            for col in cols:
                if col in means:
                    ir.series[col] = ir.series[col] * stds[col] + means[col]

        elif self.method == "minmax":
            if "minmax_mins" not in ir.metadata.extra:
                raise ValueError("No MinMax normalization stats found in metadata")

            mins = ir.metadata.extra["minmax_mins"]
            maxs = ir.metadata.extra["minmax_maxs"]

            for col in cols:
                if col in mins:
                    range_val = maxs[col] - mins[col]
                    ir.series[col] = ir.series[col] * range_val + mins[col]

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
