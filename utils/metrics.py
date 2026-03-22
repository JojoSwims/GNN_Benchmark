"""Evaluation metrics for time series forecasting."""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional boolean mask (True = include in calculation)

    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional boolean mask (True = include in calculation)

    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mask: Optional boolean mask (True = include in calculation)
        eps: Small value to prevent division by zero

    Returns:
        MAPE value as percentage (0-100 scale)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")

    # Filter out near-zero actuals to avoid division issues
    nonzero_mask = np.abs(y_true) > eps
    y_true = y_true[nonzero_mask]
    y_pred = y_pred[nonzero_mask]

    if len(y_true) == 0:
        return float("nan")

    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))
