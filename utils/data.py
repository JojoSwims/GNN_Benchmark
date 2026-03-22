"""Windowing and splitting helpers for the GNN Benchmark."""

import numpy as np


def create_sliding_windows(
    data: np.ndarray,
    input_length: int,
    horizon: int,
    y_start: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding input/target windows from a time series array.

    Args:
        data:         Array of shape ``(T, N, C)`` or ``(T, N)``.
        input_length: Number of past timesteps ``L`` in each input window.
        horizon:      Number of future timesteps ``H`` in each target window.
        y_start:      Gap between the last input step and the first target step.
                      Default ``1`` means targets start immediately after inputs.

    Returns:
        ``(x, y)`` where

        * ``x`` has shape ``(S, L, N, C)``  -- input windows
        * ``y`` has shape ``(S, H, N, C)``  -- target windows
        * ``S = T - input_length - y_start - horizon + 2``

    Raises:
        ValueError: If there is not enough data to create at least one window.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    T = data.shape[0]
    num_samples = T - input_length - y_start - horizon + 2

    if num_samples <= 0:
        raise ValueError(
            f"Not enough data for windowing. "
            f"T={T}, input_length={input_length}, horizon={horizon}, y_start={y_start}"
        )

    x_list, y_list = [], []
    for i in range(num_samples):
        x_end = i + input_length
        y_start_idx = x_end + y_start - 1
        x_list.append(data[i:x_end])
        y_list.append(data[y_start_idx : y_start_idx + horizon])

    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def split_by_time(
    data: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an array temporally into train / val / test portions.

    The split is performed on the first dimension (samples / time steps).
    Temporal order is preserved: ``train -> val -> test``.

    Args:
        data:        Array whose first dimension represents time / samples.
        train_ratio: Fraction of samples for training.
        val_ratio:   Fraction of samples for validation.

    Returns:
        ``(train, val, test)`` sub-arrays.
    """
    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]
