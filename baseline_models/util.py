import numpy as np
import pandas as pd
import h5py

def time_splits(series_or_df, train_frac=0.7, val_frac=0.1):
    """
    Works for a single-column DataFrame or a Series.
    Splits by time (no shuffling).
    """
    x = series_or_df
    n = len(x)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    train = x.iloc[:i1]
    val   = x.iloc[i1:i2]
    test  = x.iloc[i2:]
    return train, val, test

def wide2long(path):
    df=pd.read_csv(path+"/series.csv")
    df = df.fillna(0)
    #For each column c after and including the third column, make a dataframe composed of the first two columns and c
    ts_col, node_col = df.columns[:2]
    out=[]
    for c in df.columns[2:]:
        tmp = df[[ts_col, node_col, c]].copy()
        tmp.columns = [ts_col, node_col, "value"]
        # one timestamp column, then node_id columns
        wide = tmp.pivot(index=ts_col, columns=node_col, values="value").reset_index()
        out.append(wide)
    return out
    #Return the list of dataframes.


    


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
    pass