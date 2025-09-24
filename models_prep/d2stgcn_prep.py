import numpy as np
import pandas as pd
import os
"""
Some info:
This one is built very weird compared to the others:
    -Does Seq2Seq time splits
    -Keeps time splits but only one feature.
AI:Minimum you need (keep it simple)

Sliding windows with x_len, y_len, y_start and contiguous stride=1.

Calendar features (numeric): time-of-day fraction and day-of-week integer—either from DateTimeIndex or synthetically via slots_per_day. (They’re known at inference time ⇒ safe.)

Y content switch: include or drop time features (covers both clusters).

Contiguous splits: e.g., 60/20/20 or 70/10/20; no shuffling.

Save .npz: {train,val,test}.npz with x, y, x_offsets, y_offsets.

=>Technically code is written below but needs to be verified/potentially rebuilt.
--------------
For the adjacency matrix:
TODO: There is some extra stuff in there, figure out the extra stuff like direction, but at core you will eventually call
something similar to the pickle function in util
"""


def _build_time_features(T, N, index=None, add_tod=True, add_dow=True, slots_per_day=None):
    """Return (T, N, F_time) numeric features (fractional time-of-day and day-of-week)."""
    feats = []
    if add_tod:
        if index is not None:
            frac = (index.values - index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            frac = np.asarray(frac, dtype=np.float32)
        else:
            if slots_per_day is None:
                raise ValueError("slots_per_day required when no DateTimeIndex is provided.")
            frac = ((np.arange(T) % slots_per_day) / float(slots_per_day)).astype(np.float32)
        feats.append(np.tile(frac, (N, 1)).T[..., None])  # (T,N,1)

    if add_dow:
        if index is not None:
            dow = index.dayofweek.values.astype(np.float32)
        else:
            if slots_per_day is None:
                raise ValueError("slots_per_day required when no DateTimeIndex is provided.")
            dow = (((np.arange(T) // slots_per_day) % 7).astype(np.float32))
        feats.append(np.tile(dow, (N, 1)).T[..., None])  # (T,N,1)

    if not feats:
        return np.zeros((T, N, 0), dtype=np.float32)
    return np.concatenate(feats, axis=-1)

def _windowize(TNF, x_len, y_len, y_start=1):
    """Make sliding windows on (T,N,F) → X:(B,L,N,F), Y:(B,H,N,F), plus offsets."""
    T, N, F = TNF.shape
    x_offsets = np.arange(-(x_len - 1), 1)         # e.g. [-11..0]
    y_offsets = np.arange(y_start, y_start + y_len)  # e.g. [1..12]
    min_t = -x_offsets.min()
    max_t = T - y_offsets.max()  # exclusive
    X, Y = [], []
    for t in range(min_t, max_t):
        X.append(TNF[t + x_offsets, ...])
        Y.append(TNF[t + y_offsets, ...])
    return np.stack(X, 0), np.stack(Y, 0), x_offsets[:, None], y_offsets[:, None]

def prepare_seq2seq(
    data,                      # pd.DataFrame(T×N) with DateTimeIndex OR np.ndarray (T,N) or (T,N,F)
    x_len=12, y_len=12, y_start=1,
    add_tod=True, add_dow=True,
    use_index_time=False,      # True for METR/PEMS-BAY style (real timestamps); False for PEMS04/08
    slots_per_day=288,         # required if use_index_time=False and you add time features
    include_y_time=False,      # False for PEMS04/08; True for METR/PEMS-BAY
    target_channels=(0,),      # which base channels to predict, default first
    split=(0.6, 0.2, 0.2),     # (train, val, test) contiguous
):
    """Return dict with splits + offsets. No normalization, no one-hot, no pickles."""
    # 1) Coerce to (T,N,F_base) and capture index if any
    if isinstance(data, pd.DataFrame):
        idx = data.index if use_index_time else None
        base = data.values.astype(np.float32)[:, :, None]  # (T,N,1)
    else:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:  # (T,N)
            arr = arr[:, :, None]
        base = arr  # (T,N,F_base)
        idx = None

    T, N, F_base = base.shape

    # 2) Build simple time features (numeric)
    time_feats = _build_time_features(
        T, N, index=idx if use_index_time else None,
        add_tod=add_tod, add_dow=add_dow, slots_per_day=slots_per_day
    )
    full = np.concatenate([base, time_feats], axis=-1)  # (T,N,F_base+F_time)

    # 3) Sliding windows
    X_all, Y_all, x_offs, y_offs = _windowize(full, x_len, y_len, y_start)

    # 4) Control what goes into Y
    #    - PEMS04/08 style: keep only target value channels (drop time feats)
    #    - METR/PEMS-BAY style: keep target value channels + time feats
    Y_val = Y_all[..., list(target_channels)]            # (B,H,N,C_targets)
    Y = np.concatenate([Y_val, Y_all[..., F_base:]], -1) if include_y_time else Y_val
    X = X_all

    # 5) Contiguous split
    B = X.shape[0]
    n_tr = int(round(B * split[0]))
    n_va = int(round(B * split[1]))
    n_te = B - n_tr - n_va

    out = {
        "x_train": X[:n_tr],                 "y_train": Y[:n_tr],
        "x_val":   X[n_tr:n_tr+n_va],        "y_val":   Y[n_tr:n_tr+n_va],
        "x_test":  X[-n_te:] if n_te else X[0:0], "y_test":  Y[-n_te:] if n_te else Y[0:0],
        "x_offsets": x_offs, "y_offsets": y_offs,
        "F_base": F_base, "F_time": time_feats.shape[-1],
    }
    return out

def save_npz_splits(out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for cat in ("train","val","test"):
        np.savez_compressed(
            os.path.join(out_dir, f"{cat}.npz"),
            x=out[f"x_{cat}"], y=out[f"y_{cat}"],
            x_offsets=out["x_offsets"], y_offsets=out["y_offsets"]
        )




if __name__=="__main__":
    """Example use:"""
