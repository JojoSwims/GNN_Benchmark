import pandas as pd
import numpy as np
from pathlib import Path
import data_pipeline as dp

#IMPORTANT, this is the enforced node_order to ensure there is a match between
#the order of the adjacency matrix and the order of the channels
from pems_speed import METRLA_NODE_ORDER, PEMS_BAY_NODE_ORDER
from pems_volume import PEMS04_NODE_ORDER, PEMS08_NODE_ORDER
from elergone import ELERGONE_NODE_ORDER
from beijing_air import CLUSTER1_NODE_ORDER, CLUSTER2_NODE_ORDER, BEIJING_NODE_ORDER


#-----------------------Common Functions-------------------------------

def series2tensor(path):
    """Converts our intermediate CSV into a tensor shaped (T, N, C)."""
    df = pd.read_csv(path + "/series.csv")
    cols = df.columns
    ts_col, node_col = cols[0], cols[1]
    ch_cols = cols[2:]

    # Preserve original order of timestamps/nodes
    ts_order   = pd.Index(pd.unique(df[ts_col]))
    node_order = pd.Index(pd.unique(df[node_col]))

    matrices = []
    for ch in ch_cols:
        p = df.pivot(index=ts_col, columns=node_col, values=ch)
        p = p.reindex(index=ts_order, columns=node_order)
        matrices.append(p.to_numpy())  # each (T, N)

    # 👇 change is here: stack along axis=2 to get (T, N, C)
    arr = np.stack(matrices, axis=2)

    return arr

def build_index_from_data(
    data,        # numpy array of shape (T, N, C)
    L,           # input length
    H,           # forecast horizon
    y_start=1,   # 1 = predict immediately; >1 inserts a gap before Y
    train_frac=0.7,
    val_frac=0.15,
):
    """
    Return dict {'train','val','test'} with arrays of shape (num_samples, 3),
    rows are [start, split, end], where X = [start:split) and Y = [split:end).
    """
    T = int(data.shape[0])
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    assert L > 0 and H > 0 and y_start >= 1

    # The latest end time needed for a window that starts at s:
    # input covers L, then (y_start-1) gap, then H predicted steps.
    need = L + (y_start - 1) + H
    if need > T:
        raise ValueError(f"L+gap+H={need} exceeds T={T}.")

    # Convert fractions to time cutoffs (exclusive upper bounds)
    t_train_end = int(T * train_frac)
    t_val_end   = int(T * (train_frac + val_frac))

    def make_rows(a, b):
        rows = []
        # Last valid start s satisfies s + need <= b
        last_start = b - need
        for s in range(a, last_start + 1):  # stride fixed to 1
            split = s + L + (y_start - 1)   # move split to actual Y start
            end   = split + H
            rows.append([s, split, end])
        return np.asarray(rows, dtype=np.int32)

    train = make_rows(0, t_train_end)
    val   = make_rows(t_train_end, t_val_end)
    test  = make_rows(t_val_end, T)

    return {"train": train, "val": val, "test": test}

#-----------------------STAEFormer-------------------------------

def save_index_npz(path, index_dict):
    np.savez(f"{path.rstrip('/')}/index.npz",
             train=index_dict["train"], val=index_dict["val"], test=index_dict["test"])

def save_data_npz(path, data):
    """
    Save the data array (T, N, C) to `data.npz` under key 'data'.
    """
    out_path = f"{path.rstrip('/')}/data.npz"
    np.savez(out_path, data=data)


if __name__=="__main__":
    print("here")
    PATH="../temp/pemsbay" #Choose the dataset here
    train_ratio=0.7
    H=12
    L=12
    val_ratio=0.1
    TARGET_DIR="./" #Output path for the npz files

#-----------------------Generating Required Files for STAEFormer-------------------------------

    mask=dp.fill_zeroes(PATH)
    
    data = series2tensor(PATH)
    index_dict = build_index_from_data(
            data=data,
            L=L,
            H=H,
            y_start=1,
            train_frac=train_ratio,
            val_frac=val_ratio,
        )

    save_data_npz(TARGET_DIR, data)
    save_index_npz(TARGET_DIR, index_dict)

#-----------------------Generating Required Files for STAEFormer-------------------------------

