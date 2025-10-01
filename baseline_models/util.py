import numpy as np
import pandas as pd
import h5py


def load_csv(path):
    """Load a CSV file containing sensor readings indexed by timestamp.

    Parameters
    ----------
    path : str or PathLike
        Location of the CSV file. The file is expected to have an unnamed
        timestamp column that will become the index.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape ``(T, N)`` where ``T`` is the number of time steps
        and ``N`` the number of sensors. The index is parsed as timestamps and
        all values are converted to ``float`` with missing entries filled with
        ``0``.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Convert everything to floats
    df = df.astype(float)

    # Ensure missing values are treated consistently with the zero null value.
    df = df.copy()
    df = df.fillna(0)

    return df  # shape: (T, N) with DateTimeIndex(freq='5T')


def load_h5(path):
    """Load an HDF5 file containing sensor data.

    Parameters
    ----------
    path : str or PathLike
        Location of the ``.h5`` file to load.

    Returns
    -------
    h5py.File
        Handle to the opened HDF5 file. The caller is responsible for closing
        the file when finished.
    """
    return h5py.File(path, "r")


def _resolve_mask(labels, mask=None):
    """Compute a validity mask for label arrays.

    Missing targets are assumed to be encoded as ``0``. Any provided mask is
    broadcast to match ``labels`` and returned directly.

    Parameters
    ----------
    labels : numpy.ndarray
        Reference array used to infer valid positions.
    mask : numpy.ndarray, optional
        Pre-computed boolean mask identifying valid entries. If provided it is
        broadcast to ``labels`` and converted to ``bool``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of valid entries.
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape:
            mask = np.broadcast_to(mask, labels.shape)
        return mask

    return np.asarray(labels) != 0


def _masked_mean(values, mask):
    """Return the mean of ``values`` constrained by ``mask``.

    Parameters
    ----------
    values : numpy.ndarray
        Array containing the values to average.
    mask : numpy.ndarray or None
        Boolean mask indicating valid entries.

    Returns
    -------
    float
        Mean of ``values`` over valid entries or ``np.nan`` when no valid
        entries exist.
    """
    if mask is None:
        return float(np.nanmean(values))

    masked_values = values[mask]
    if masked_values.size == 0:
        return float("nan")
    return float(np.nanmean(masked_values))


def mae(y_true, y_pred, mask=None):
    """Compute the mean absolute error between predictions and targets.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth values.
    y_pred : numpy.ndarray
        Predicted values.
    mask : numpy.ndarray, optional
        Boolean array selecting entries to include in the computation. When not
        provided, entries equal to ``0`` in ``y_true`` are treated as missing
        data and ignored. Any NaNs in the inputs should already be replaced
        with ``0`` when loading the data.

    Returns
    -------
    float
        Mean absolute error computed over valid entries.
    """
    validity_mask = _resolve_mask(y_true, mask=mask)
    errors = np.abs(y_true - y_pred)
    return _masked_mean(errors, validity_mask)


def rmse(y_true, y_pred, mask=None):
    """Compute the root mean squared error between predictions and targets.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth values.
    y_pred : numpy.ndarray
        Predicted values.
    mask : numpy.ndarray, optional
        Boolean array selecting entries to include in the computation. When not
        provided, entries equal to ``0`` in ``y_true`` are treated as missing
        data and ignored. Any NaNs in the inputs should already be replaced
        with ``0`` when loading the data.

    Returns
    -------
    float
        Root mean squared error computed over valid entries.
    """
    validity_mask = _resolve_mask(y_true, mask=mask)
    squared_errors = (y_true - y_pred) ** 2
    mse = _masked_mean(squared_errors, validity_mask)
    return float(np.sqrt(mse)) if np.isfinite(mse) else mse


def mape(y_true, y_pred, mask=None, eps=1e-6):
    """Compute the mean absolute percentage error between two arrays.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth values.
    y_pred : numpy.ndarray
        Predicted values.
    mask : numpy.ndarray, optional
        Boolean array selecting entries to include in the computation. When not
        provided, entries equal to ``0`` in ``y_true`` are treated as missing
        data and ignored. Any NaNs in the inputs should already be replaced
        with ``0`` when loading the data.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        Mean absolute percentage error in percentage points computed over valid
        entries.
    """
    validity_mask = _resolve_mask(y_true, mask=mask)
    denom = np.clip(np.abs(y_true), eps, None)
    percentage_errors = np.abs((y_true - y_pred) / denom) * 100
    return _masked_mean(percentage_errors, validity_mask)

if __name__=="__main__":

    path = "./pemsbay/pemsbay.csv"

    df = load_csv(path)
    total_entries = df.size
    zero_count = (df == 0).sum().sum()
    zero_pct = zero_count / total_entries * 100

    print(f"Total zeros: {zero_count} ({zero_pct:.2f}% of all values)")
