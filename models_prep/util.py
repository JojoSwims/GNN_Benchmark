    
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

#TODO: Understand this function
#TODO: Add something related to symmetric entries for undirected graphs!!!
#TODO: There is a problem with this function. If a node has no edges, it will be absent.
#This is something that needs to be fixed, potentially by getting the node list from the metadata as part of the prep process.
def edges_csv_to_adj(edges_csv: str, pkl_path: str):
    """Create adj_mx.pkl from edges.csv."""
    edges = pd.read_csv(edges_csv)
    #Ensures our columns have the correct names
    edges.rename(columns={edges.columns[0]: "src", edges.columns[1]: "dst", **({edges.columns[2]: "weight"} if edges.shape[1] >= 3 else {})}, inplace=True)

    node_ids = sorted(pd.unique(edges[["src", "dst"]].to_numpy().ravel()))
    id2ind = {nid: i for i, nid in enumerate(node_ids)}
    adj = np.zeros((len(node_ids), len(node_ids)), dtype=float)

    if "weight" in edges.columns:
        for src, dst, w in edges[["src", "dst", "weight"]].to_numpy():
            adj[id2ind[src], id2ind[dst]] = w
    else:
        for src, dst in edges[["src", "dst"]].to_numpy():
            adj[id2ind[src], id2ind[dst]] = 1.0

    #TODO: Make sure we expect a diagonal with 1.0 and not some other value
    np.fill_diagonal(adj, 1.0)
    with open(pkl_path, "wb") as f:
        pickle.dump((node_ids, id2ind, adj), f)


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