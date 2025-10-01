import numpy as np
import pandas as pd
import h5py



def load_csv(path):
    """
    Expects a CSV like:
    ,773869,767541,767542,717447
    2012-03-01 00:00:00,64.375,67.625,67.125,61.5
    2012-03-01 00:05:00,62.666668,68.55556,65.44444,62.444443
    where the first (unnamed) column is the timestamp.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Convert everything to floats
    df = df.astype(float)

    return df  # shape: (T, N) with DateTimeIndex(freq='5T')

def load_h5(path):
    pass

def mae(y_true, y_pred):
    return float(np.nanmean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = (y_true != 0) & ~np.isnan(y_true)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _mask(labels, null_val):
    if np.isnan(null_val):
        m = ~np.isnan(labels)
    else:
        m = labels != null_val
    return m

def masked_mae_np(preds, labels, null_val=np.nan):
    m = _mask(labels, null_val)
    if m.sum() == 0:
        return np.nan
    return np.abs(preds - labels)[m].mean()

def masked_mse_np(preds, labels, null_val=np.nan):
    m = _mask(labels, null_val)
    if m.sum() == 0:
        return np.nan
    return ((preds - labels)**2)[m].mean()

def masked_rmse_np(preds, labels, null_val=np.nan):
    val = masked_mse_np(preds, labels, null_val)
    return np.sqrt(val) if np.isfinite(val) else val

def masked_mape_np(preds, labels, null_val=np.nan, eps=1e-6):
    m = _mask(labels, null_val)
    if m.sum() == 0:
        return np.nan
    denom = np.clip(np.abs(labels), eps, None)
    return (np.abs((preds - labels)/denom)[m]).mean()

if __name__=="__main__":

    path = "./pemsbay/pemsbay.csv"

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    total_entries = df.size
    zero_count = (df == 0).sum().sum()
    zero_pct = zero_count / total_entries * 100

    print(f"Total zeros: {zero_count} ({zero_pct:.2f}% of all values)")
