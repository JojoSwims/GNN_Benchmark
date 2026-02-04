"""
Transforms for intermediate representation manipulation.

This module provides transforms that modify the IntermediateRepresentation
in-place. Transforms are applied using `ir.apply(transform)` and their
descriptions are recorded in the transform history.

Classes:
    Transform: Abstract base class for implementing custom transforms.

    Imputation transforms (handle missing values):
        FillZeros: Replace NaN with zeros and generate observation mask.
        ForwardFill: Forward fill within each node's time series.
        KalmanImpute: Kalman filter with seasonal components (requires statsmodels).

    Temporal transforms (time-based operations):
        AddTimeFeatures: Add time_of_day, day_of_week, time_of_year columns.
        FilterHours: Keep only specific hours of the day.
        FilterDateRange: Keep only data within a date range.
        Resample: Resample to a different frequency (mean, sum, etc.).

    Normalization transforms (feature scaling):
        ZScoreNormalize: Z-score normalization using training statistics.
        MinMaxNormalize: Min-max normalization to [0, 1] range.
        Denormalize: Reverse normalization using stored statistics.

Example:
    >>> from gnn_benchmark.transforms import FillZeros, AddTimeFeatures, ZScoreNormalize
    >>> ir.apply(FillZeros())
    >>> ir.apply(AddTimeFeatures(time_of_day=True, day_of_week=True))
    >>> ir.apply(ZScoreNormalize(train_end="2015-01-01"))
    >>> print(ir.metadata.transform_history)
"""

from gnn_benchmark.transforms.base import Transform
from gnn_benchmark.transforms.imputation import FillZeros, ForwardFill, KalmanImpute
from gnn_benchmark.transforms.temporal import (
    AddTimeFeatures,
    FilterHours,
    FilterDateRange,
    Resample,
)
from gnn_benchmark.transforms.normalization import (
    ZScoreNormalize,
    MinMaxNormalize,
    Denormalize,
)

__all__ = [
    "Transform",
    # Imputation
    "FillZeros",
    "ForwardFill",
    "KalmanImpute",
    # Temporal
    "AddTimeFeatures",
    "FilterHours",
    "FilterDateRange",
    "Resample",
    # Normalization
    "ZScoreNormalize",
    "MinMaxNormalize",
    "Denormalize",
]
