"""
Windowing helpers and DataLoader builder for the GNN Benchmark.

This module exposes:
- ``create_sliding_windows``: Convert a (T, N, C) array into (S, L, N, C) / (S, H, N, C) pairs.
- ``split_by_time``: Split the first dimension of an array into train / val / test.
- ``make_dataloaders``: Wrap numpy arrays into PyTorch DataLoaders ready for ``BenchmarkModel``.

``ModelExporter`` delegates its helper methods to these functions so that
the benchmark runner can reuse the same logic without going through an exporter.
"""

from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from torch.utils.data import DataLoader  # noqa: F811


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

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

        * ``x`` has shape ``(S, L, N, C)``  — input windows
        * ``y`` has shape ``(S, H, N, C)``  — target windows
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
    Temporal order is preserved: ``train → val → test``.

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


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def make_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> "tuple[DataLoader, DataLoader, DataLoader]":
    """
    Wrap pre-windowed numpy arrays into PyTorch DataLoaders.

    All arrays are cast to ``float32`` before wrapping.  The test loader is
    **always** created with ``shuffle=False`` to guarantee that prediction
    rows align with ground-truth rows in the same order.

    Args:
        x_train:       Training inputs  ``[S_train, L, N, D_in]``.
        y_train:       Training targets ``[S_train, H, N, D_out]``.
        x_val:         Validation inputs  ``[S_val, L, N, D_in]``.
        y_val:         Validation targets ``[S_val, H, N, D_out]``.
        x_test:        Test inputs  ``[S_test, L, N, D_in]``.
        y_test:        Test targets ``[S_test, H, N, D_out]``.
        batch_size:    Batch size for all loaders.
        shuffle_train: Whether to shuffle the training loader (default ``True``).

    Returns:
        ``(train_loader, val_loader, test_loader)``

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for make_dataloaders. "
            "Install it with: pip install torch"
        )

    def _to_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> "DataLoader":
        tx = torch.from_numpy(x.astype(np.float32))
        ty = torch.from_numpy(y.astype(np.float32))
        return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(x_train, y_train, shuffle=shuffle_train)
    val_loader   = _to_loader(x_val,   y_val,   shuffle=False)
    test_loader  = _to_loader(x_test,  y_test,  shuffle=False)

    return train_loader, val_loader, test_loader
