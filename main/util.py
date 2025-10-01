import pandas as pd
import numpy as np








#----------Functions acting on series.csv---------------

def series2tensor(path):
    """Converts a file with our intermediate representation into a (T,C,N) tensor"""
    df=pd.read_csv(path+"/series.csv")
    cols = df.columns
    ts_col, node_col = cols[0], cols[1]
    ch_cols = cols[2:]

    #Keeps track of order and allows us to preserve order.
    ts_order   = pd.Index(pd.unique(df[ts_col]))
    node_order = pd.Index(pd.unique(df[node_col]))

    matrices = []
    for ch in ch_cols:
        p = df.pivot(index=ts_col, columns=node_col, values=ch)
        p = p.reindex(index=ts_order, columns=node_order)
        matrices.append(p.to_numpy())
    arr = np.stack(matrices, axis=1)  # (T, C, N)
    return arr

def split_tcn(arr, train=0.7, val=0.1, test=0.2):
    """Split (T, C, N) into contiguous train/val/test along time."""
    T = arr.shape[0]
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("train + val + test must equal 1.0")

    n_train = int(T * train)
    n_val   = int(T * val)

    train_arr = arr[:n_train]
    val_arr   = arr[n_train:n_train + n_val]
    test_arr  = arr[n_train + n_val:]

    return train_arr, val_arr, test_arr

def zscore(train, val=None, test=None, eps=1e-8):
    """
    train, val, test: arrays shaped (T, C, N)
    """
    mu    = train.mean(axis=0, keepdims=True)             # (1, C, N)
    sigma = train.std(axis=0, ddof=0, keepdims=True)      # (1, C, N)
    sigma = np.where(sigma == 0, 1.0, sigma)              # avoid /0

    train_std = (train - mu) / (sigma + eps)
    val_std   = (val - mu) / (sigma + eps) if val is not None else None
    test_std  = (test - mu) / (sigma + eps) if test is not None else None
    return train_std, val_std, test_std

def get_node_list(path):
    """Get a list of individual nodes from our series.csv file"""
    df=pd.read_csv(path+"/series.csv")
    #Get all unique values from the second column:
    nodes=pd.unique(df.iloc[:, 1])
    return nodes

#-----------Functions acting on edges.csv-----------------
def exp_decay_weights(path, r, tau):
    df=pd.read_csv(path+"/edges.csv") #Our from, to, cost edges dataframe
    nodes=get_node_list(path) #Get a list of all the node ids.
    d = df.iloc[:, 2].to_numpy(dtype=float)
    w = np.exp(-d / tau) * (d <= r)
    out = df
    out.iloc[:, 2] = w
    return out

def edges_to_np_array(path, dataset_name):
    """Takes our from, to, cost format adjacency matrix and turns it into an nxn array (for n nodes)"""
    if dataset_name in ["metrla", "pemsbay"]:
        symmetric=False
    elif dataset_name in ["PEMS08", "PEMS04", "aqi"]:
        symmetric=True
    else:
        raise NotImplementedError("Dataset name is not in supported Datasets list")
    df=pd.read_csv(path+"/edges.csv") #Our from, to, cost edges dataframe
    nodes=get_node_list(path) #Get a list of all the node ids.



    src_col = df.columns[0]
    dst_col = df.columns[1]
    cost_col = df.columns[2]

    A = (
        df.pivot_table(index=src_col, columns=dst_col, values=cost_col, fill_value=0.0)
          .reindex(index=nodes, columns=nodes, fill_value=0.0)
          .to_numpy(dtype=float)
    )
    
    if symmetric:
        print(dataset_name)
        A = np.maximum(A, A.T)

    return A

def flatten_weights(np_adj):
    """
    Takes an adjacency matrix in numpy format and transform
    """
    np_adj = (np_adj != 0).astype(float)
    return np_adj

#This function was AI generated for time-saving measures:
def is_symmetric(A, atol=1e-12, rtol=0.0, equal_nan=False):
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return False
        return np.allclose(A, A.T, atol=atol, rtol=rtol, equal_nan=equal_nan)

def get_degree_matrix(A):
    """
    Takes an adjacency matrix and in numpy format and gets the degree matrix D
    """
    if not is_symmetric(A):
        raise NotImplementedError("A matrix for a directed graph was passed to this function. This function is only implemented for symmetric matrices representing undirected graphs.")
    
    A=flatten_weights(A)
    deg = A.sum(axis=0)  

    D   = np.diag(deg)
    return D

def _deg_vec(D):
    """Accept D as diagonal matrix or 1-D degree vector; return degree vector."""
    D = np.asarray(D)
    return np.diag(D) if D.ndim == 2 else D

def norm_adj_symmetric(A, D):
    """
    Â_sym = D^{-1/2} (A + I) D^{-1/2}
    """
    n = A.shape[0]
    d = _deg_vec(D)
    dmh = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_hat = A + np.eye(n)
    return dmh[:, None] * A_hat * dmh[None, :]

def random_walk_matrix(A, D):
    """
    P_rw = D^{-1} A   (row-normalized / out-normalized)
    """
    d = _deg_vec(D)
    dinv = np.where(d > 0, 1.0 / d, 0.0)
    return dinv[:, None] * A

def normalized_laplacian(A, D):
    """
    L = I - D^{-1/2} A D^{-1/2}
    """
    n = A.shape[0]
    d = _deg_vec(D)
    dmh = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    return np.eye(n) - (dmh[:, None] * A * dmh[None, :])


if __name__=="__main__":
    paths=["../temp/pemsbay", "../temp/metrla", "../temp/aqi", "../temp/PEMS04", "../temp/PEMS08"]

    for p in paths:
        print(p)
        A=edges_to_np_array(p, p.removeprefix("../temp/"))
        print(is_symmetric(A))

