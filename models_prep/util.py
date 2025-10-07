
import pandas as pd
import numpy as np
import pickle
import csv
from pathlib import Path

FIRST_COLUMN_NAME="value1"


def series_csv_to_h5(series_csv: str, h5_path: str):
    """Convert series.csv into the HDF5 format expected by generate_training_data.py.
    
    Parameters
    ----------
    series_csv : str
        Path to the input CSV file.
    h5_path : str
        Path to the output HDF5 file.
    value_col : str or None
        Name of the value column to pivot. If None, use all columns except 'ts' and 'node_id'.
    """
    df = pd.read_csv(series_csv, parse_dates=["ts"])
    df.rename(columns={df.columns[0]: "ts", df.columns[1]: "node_id"}, inplace=True)


    value_cols = [c for c in df.columns if c not in ("ts", "node_id")]
    table = df.pivot(index="ts", columns="node_id", values=value_cols)
    
    table.to_hdf(h5_path, key="df")

def get_node_list(path):
    """Get a list of individual nodes from our series.csv file"""
    df=pd.read_csv(path+"/series.csv")
    #Get all unique values from the second column:
    nodes=pd.unique(df.iloc[:, 1])
    return nodes

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
    np.fill_diagonal(A, 1.0)

    return A, nodes

def edges_csv_to_adj(path: str, pkl_path: str, datataset_name):
    """Create adj_mx.pkl from edges.csv.
    IMPORTANT: path is the path to the folder containing all the dataset's intermediate representation.
    Dataset name is the name of the folder ex: metrla
    """
    adj, nodes=edges_to_np_array(path, datataset_name)
    id2ind = {nid: i for i, nid in enumerate(nodes)}

    np.fill_diagonal(adj, 1.0)
    with open(pkl_path, "wb") as f:
        pickle.dump((nodes, id2ind, adj), f)
    return adj


def write_sensor_ids(edges_csv: str, output_txt: str):

    """
    Read an edges.csv file and write the unique sensor IDs to output_txt.
    Parameters
    ----------
    edges_csv : str
        Path to the CSV containing 'src' and 'dst' columns.
    output_txt : str
        Path (as a string) to the text file that will store one sensor ID per line.
    """
    sensor_ids = set()

    # Collect all src/dst IDs
    with open(edges_csv, newline="") as f:
        reader = csv.DictReader(f)
        #Ensure the names are correct
        reader.fieldnames = ["src", "dst"] + reader.fieldnames[2:]

        for row in reader:
            sensor_ids.add(row["src"])
            sensor_ids.add(row["dst"])

    # Sort for stable ordering and write to file
    sensor_list = sorted(sensor_ids)
    Path(output_txt).write_text(",".join(sensor_list))


def series_csv_to_npz(series_csv, npz_path, feature_cols: list[str] | None = None, dtype=np.float32):
    """
    Convert series.csv (columns: ts, node_id, value1, [value2, ...]) into an NPZ:
      - data: (T, N, F) array
      - timestamps: (T,) datetime64[ns]
      - node_ids: (N,) object (strings)
      - feature_names: (F,) object

    Assumptions: clean rows (no duplicate (ts, node_id)); missing entries become NaN.
    """
    # Load CSV
    df = pd.read_csv(series_csv, parse_dates=["ts"])
    df.rename(columns={df.columns[0]: "ts", df.columns[1]: "node_id"}, inplace=True)


    # Pick features: all columns except ts/node_id if not provided
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("ts", "node_id")]
    if not feature_cols:
        raise ValueError("No feature columns found (need columns beyond ts and node_id).")

    # Keep only needed columns, ensure types
    df = df[["ts", "node_id"] + feature_cols].copy()
    df["node_id"] = df["node_id"].astype(str)

    # Sort by time (only)
    df.sort_values("ts", inplace=True)

    # Build consistent axes
    # - timestamps: union of all timestamps, ascending
    all_ts = sorted(df["ts"].unique())
    # - node_ids: first-seen order (no need to sort)
    node_ids = list(dict.fromkeys(df["node_id"]))

    # For each feature, make a (T, N) table and align to the common axes
    planes = []
    for feat in feature_cols:
        tmp = df[["ts", "node_id", feat]]
        wide = tmp.pivot(index="ts", columns="node_id", values=feat)
        # align to the full axes
        wide = wide.reindex(index=all_ts, columns=node_ids)
        planes.append(wide.to_numpy(dtype=dtype))  # (T, N)

    # Stack features -> (T, N, F)
    data = np.stack(planes, axis=-1)

    # Save
    timestamps = pd.Index(all_ts).values.astype("datetime64[ns]")
    feature_names = np.array(feature_cols, dtype=object)
    node_ids = np.array(node_ids, dtype=object)

    np.savez_compressed(
        npz_path,
        data=data,
        timestamps=timestamps,
        node_ids=node_ids,
        feature_names=feature_names,
    )

if __name__=="__main__":
    print(edges_csv_to_adj("../temp/metrla", "here", "metrla").shape)