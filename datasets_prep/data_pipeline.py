import pandas as pd
import numpy as np


#TODO: Persistence as an imputation method.

def series2tensor(path, impute:bool=False):
    """Converts a file with our intermediate representation into a (T,C,N) tensor"""
    mask=None
    df=pd.read_csv(path+"/series.csv")
    cols = df.columns
    ts_col, node_col = cols[0], cols[1]
    ch_cols = cols[2:]

    # Print the columns that become channels (C)
    print("Channel columns (C):", list(ch_cols))

    #Keeps track of order and allows us to preserve order.
    ts_order   = pd.Index(pd.unique(df[ts_col]))
    node_order = pd.Index(pd.unique(df[node_col]))

    matrices = []
    for ch in ch_cols:
        p = df.pivot(index=ts_col, columns=node_col, values=ch)
        p = p.reindex(index=ts_order, columns=node_order)
        matrices.append(p.to_numpy())
    arr = np.stack(matrices, axis=1)  # (T, C, N)
    return arr, mask



def add_time_channels(path, tod : bool =True, tow : bool = True, toy:bool = False):
    df = pd.read_csv(path+"/series.csv", parse_dates=[0])
    ts_col = df.columns[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce")

    if tod:
        df["time_of_day"] = 0.0
    if tow:
        df["day_of_week"] = 0  # int; will overwrite below
    if toy:
        df["time_of_year"] = 0.0

    if tod:
        secs = (
            ts.dt.hour.astype("int64") * 3600
            + ts.dt.minute.astype("int64") * 60
            + ts.dt.second.astype("int64")
        )
        df["time_of_day"] = secs / 86400.0  # [0,1)
    if tow:
        df["day_of_week"] = ts.dt.dayofweek.astype("int8")  # 0=Mon .. 6=Sun

    if toy:
        # day-of-year (1..365/366) + fractional day, normalized by year length
        doy = ts.dt.dayofyear.astype("int64")
        frac_day = df["time_of_day"] if tod else (
            (ts.dt.hour*3600 + ts.dt.minute*60 + ts.dt.second) / 86400.0
        )
        is_leap = ts.dt.is_leap_year
        year_len = np.where(is_leap.to_numpy(), 366.0, 365.0)
        df["time_of_year"] = ((doy - 1) + frac_day.to_numpy()) / year_len  # [0,1)
    df.to_csv(path+"/series.csv", index=False)
    return df

def get_split_timestamps(path: str, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    Assumes df.columns[0] is the timestamp column.
    Returns (t_train_end, t_val_end), both as pandas Timestamps.

    Inclusivity:
      - Train: timestamp <= t_train_end (inclusive)
      - Val:   t_train_end < timestamp <= t_val_end (start exclusive, end inclusive)
      - Test:  timestamp > t_val_end (exclusive)
    """
    df = pd.read_csv(path+"/series.csv")
    ts_col = df.columns[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values().unique()
    n = len(ts)
    if n == 0:
        raise ValueError("No valid timestamps found in the first column.")

    cut_train = int(round(n * train_ratio))
    cut_val   = int(round(n * (train_ratio + val_ratio)))

    # clamp to valid range and monotone
    cut_train = max(1, min(cut_train, n))
    cut_val   = max(cut_train, min(cut_val, n))

    t_train_end = pd.Timestamp(ts[cut_train - 1])
    t_val_end   = pd.Timestamp(ts[cut_val - 1])
    return t_train_end, t_val_end


#------------Windowization------------------------

def windowize(path, input_cols: list[int], target_cols: list[int], L: int, H: int, y_start: int = 1):
    """
    df columns: [ts, node, v1, v2, ...]
    input_cols/target_cols: ints where 0 -> v1 (3rd column), 1 -> v2, ...
    Returns:
      x: (S, L, N, F_in)
      y: (S, H, N, F_y)
    S = num_samples = number of sliding windows we generated.
    L = input_length = how many past steps in each input window.
    N = num_nodes = number of sensors/nodes (one per column in the wide pivot).
    F_in = input_dim = number of input feature channels you included (size of input_cols).
    """
    df=pd.read_csv(path+"/series.csv")
    ts_col, node_col = df.columns[0], df.columns[1]
    vals = list(df.columns[2:])

    # map indices to names
    def pick(cols_idx):
        return [vals[i] for i in cols_idx]

    in_names  = pick(input_cols)
    tgt_names = pick(target_cols)

    # sort + get grids
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.sort_values([ts_col, node_col]).reset_index(drop=True)
    timestamps = pd.Index(sorted(d[ts_col].dropna().unique()))
    nodes      = pd.Index(sorted(d[node_col].dropna().unique()))

    # pivot helper -> (T, N, F)
    def to_array(names):
        mats = []
        for c in names:
            w = (d[[ts_col, node_col, c]]
                 .pivot(index=ts_col, columns=node_col, values=c)
                 .reindex(index=timestamps, columns=nodes))
            mats.append(w.to_numpy(dtype=float))
        return np.stack(mats, axis=-1)  # (T, N, F)

    X_all = to_array(in_names)   # (T, N, F_in)
    Y_all = to_array(tgt_names)  # (T, N, F_y)
    T, N, F_in = X_all.shape
    F_y = Y_all.shape[-1]

    # window loop
    min_t = L - 1
    max_t = T - (y_start + H - 1)  # exclusive
    S = max(0, max_t - min_t)

    xs, ys = [], []
    for t in range(min_t, max_t):
        xs.append(X_all[t-(L-1):t+1, :, :])                          # (L, N, F_in)
        ys.append(Y_all[t+y_start : t+y_start+H, :, :])              # (H, N, F_y)

    x = np.stack(xs, axis=0) if S > 0 else np.empty((0, L, N, F_in))
    y = np.stack(ys, axis=0) if S > 0 else np.empty((0, H, N, F_y))
    return x, y

def zscore(path: str, channels, t_train_end: pd.Timestamp, t_val_end: pd.Timestamp, eps: float = 1e-8):
    """
    Schema: [timestamp, node, value1, value2, ...]
    - Compute mu/sigma on TRAIN ONLY (rows with timestamp <= t_train_end).
    - Apply z-score to WHOLE DataFrame for selected channels.
    - Return (df_norm, train_df, val_df, test_df).

    Inclusivity:
      - Train: timestamp <= t_train_end      (inclusive)
      - Val:   t_train_end < ts <= t_val_end (exclusive, inclusive)
      - Test:  timestamp > t_val_end         (exclusive)
    """
    out = pd.read_csv(path+"/series.csv")
    time_col = out.columns[0]

    # ensure datetime + sort
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values(time_col).reset_index(drop=True)

    # pick value columns (ignore first two)
    value_cols = list(out.columns[2:])

    norm_cols = []
    for ch in channels:
        norm_cols.append(value_cols[ch])
    
    # train mask (by time)
    train_mask = out[time_col] <= t_train_end

    # mu/sigma on TRAIN ONLY (NaN-safe); apply to WHOLE df
    X_train = out.loc[train_mask, norm_cols].to_numpy(dtype=float)
    mu = np.nanmean(X_train, axis=0, keepdims=True)
    sigma = np.nanstd(X_train, axis=0, ddof=0, keepdims=True)
    sigma = np.where((sigma == 0) | ~np.isfinite(sigma), 1.0, sigma)

    X_all = out[norm_cols].to_numpy(dtype=float)
    out.loc[:, norm_cols] = (X_all - mu) / (sigma + eps)

    # splits (normalized)
    train_df = out[out[time_col] <= t_train_end]
    val_df   = out[(out[time_col] > t_train_end) & (out[time_col] <= t_val_end)]
    test_df  = out[out[time_col] > t_val_end]

    out.to_csv(path+"/series.csv", index=False)
    return out, mu, sigma

#--------------------Fill values------------------------------

def fill_zeroes(path):

    out = pd.read_csv(path+"/series.csv")
    mask =out.notna()
    out = out.fillna(0)
    out.to_csv(path+"/series.csv", index=False)

    return mask

#-------------------------------------------------------------


def split_tensor(tensor, train_split, val_split):
    T = tensor.shape[0]

    t_tr = int(round(T * train_split))
    t_va = int(round(T * val_split))

    # Clip so we don't exceed T
    cut1 = max(0, min(T, t_tr))
    cut2 = max(cut1, min(T, cut1 + t_va))

    train = tensor[:cut1]
    val   = tensor[cut1:cut2]
    test  = tensor[cut2:]

    return train, val, test

'''
def zscore(train, val=None, test=None, channels=None, eps=1e-8):
    """
    Standardize selected channels of (T, C, N) using TRAIN stats.
    Per-(channel, node) normalization across time (axis=0).
    Returns NEW arrays (inputs are not modified).

    Args:
        train, val, test : np.ndarray shaped (T, C, N)
        channels         : list/array of channel indices
                           None => all channels.
        eps              : small value to avoid divide-by-zero.

    Returns:
        train_std, val_std, test_std
    """
    if train.ndim != 3:
        raise ValueError("Expected (T, C, N) arrays.")

    T, C, N = train.shape

    # Normalize `channels` to integer indices
    if channels is None:
        ch_idx = np.arange(C)
    else:
        arr = np.asarray(channels)
        ch_idx = np.unique(arr.astype(int))
        if ch_idx.size and (ch_idx.min() < 0 or ch_idx.max() >= C):
            raise IndexError(f"Channel indices must be in [0, {C-1}]")

    tr = train.copy()
    va = val.copy()   if val  is not None else None
    te = test.copy()  if test is not None else None

    for c in ch_idx:
        # Per-node stats over time from TRAIN ONLY
        mu    = np.nanmean(train[:, c, :], axis=0, keepdims=True)  # (1, N)
        sigma = np.nanstd( train[:, c, :], axis=0, keepdims=True)  # (1, N)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

        tr[:, c, :] = (tr[:, c, :] - mu) / (sigma + eps)
        if va is not None:
            va[:, c, :] = (va[:, c, :] - mu) / (sigma + eps)
        if te is not None:
            te[:, c, :] = (te[:, c, :] - mu) / (sigma + eps)

    return tr, va, te
'''

if __name__=="__main__":
    PATH="../temp/metrla"
    #ALL FUNCTIONS BELOW MODIFY series.csv directly.

    #Add time channels (the graph wave net format)
    add_time_channels(PATH, tod=True, tow=True)

    #Get our train val timestamps, (used to split in later functions)
    train_ratio=0.7
    val_ratio=0.1
    t_train_end, t_val_end = get_split_timestamps(PATH, train_ratio, val_ratio)

    #zscore
    zscore(PATH, [0], t_train_end, t_val_end)

    #Fill in NaNs with something, either all zeroes or some smart_fill (to be implemented this weekend)
    mask=fill_zeroes(PATH) #Can run windowize on this, need



    #Generate the train test val
    x, y=windowize(PATH,[0,1,2],[0],12, 12)
    x_train, x_test, x_val=split_tensor(x, train_ratio, val_ratio)
    x_train, x_test, x_val=split_tensor(y, train_ratio, val_ratio)

    #Can save these to disk with npz, then load them back when running the model


    

