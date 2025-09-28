import pandas as pd
from pathlib import Path
import numpy as np
import gdown
import pickle
import shutil
import urllib.request

BAY_ID="1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"
METRLA_ID="1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
BAY_URL="https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl"
METRLA_URL="https://raw.githubusercontent.com/chnsh/DCRNN/master/data/sensor_graph/adj_mx.pkl"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/pems_speed")
default_set=[]

def _download(dataset_select, zip_path = ZIP_DEFAULT_PATH, out_dir = OUT_DEFAULT_PATH):
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pemsbay" in dataset_select:
        gdown.download(id=BAY_ID, output=str(out_dir/"pems-bay.h5"), quiet=False)
        urllib.request.urlretrieve(BAY_URL, out_dir/"adj_mx_bay.pkl")

    if "metrla" in dataset_select:
        gdown.download(id=METRLA_ID, output=str(out_dir/"metr-la.h5"), quiet=False)
        urllib.request.urlretrieve(METRLA_URL, out_dir/"adj_mx.pkl")



def _cleanup(target_dir):
    """Delete files (not dirs) inside target_dir, move remaining items up one level, then remove target_dir."""
    p = Path(target_dir)
    if not p.exists():
        return
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    # Delete files and symlinks directly under target_dir
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()

    # Move remaining entries (typically directories) to the parent
    parent = p.parent
    for child in list(p.iterdir()):
        dest = parent / child.name
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        shutil.move(str(child), str(dest))

    # 3) Remove now-empty target_dir
    p.rmdir()
    pass

def _individual_prepare_data(out_dir, dataset):
    if dataset=="metrla":
        fpath=out_dir/"metr-la.h5"
    elif dataset=="pemsbay":
        fpath=out_dir/"pems-bay.h5"
    else:
        raise NotImplementedError("An unaccepted dataset_select value was passed to prepare(), please pass a list containing one or both of 'metrla' and 'pemsbay")
        
    with pd.HDFStore(fpath, mode="r") as store:
        key = store.keys()[0]           # Here, '/df'
        wide = store.get(key)            # Is already a pandas dataframe.
    
    wide.index.name = "ts"
    long = (
        wide.stack()
            .rename_axis(["ts", "node_id"])
            .reset_index(name="value")
    )
    return long

def _individual_prepare_adj(out_dir, dataset):
    if dataset=="metrla":
        fpath=out_dir/"adj_mx.pkl"
    elif dataset=="pemsbay":
        fpath=out_dir/"adj_mx_bay.pkl"
    else:
        raise NotImplementedError("An unaccepted dataset_select value was passed to prepare(), please pass a list containing one or both of 'metrla' and 'pemsbay")
    with open(fpath, "rb") as f:
        obj = pickle.load(f, encoding="latin1")    # file object, not a string
    ids, id_to_ind, adj = obj[0], obj[1], np.asarray(obj[2])
    i, j = np.nonzero(adj); m = i != j
    edges = pd.DataFrame({"to":[ids[b] for b in j[m]],
                        "from":[ids[a] for a in i[m]],
                        "cost":adj[i[m], j[m]].astype(float)})
    return edges

def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH, dataset_select: list =["pemsbay", "metrla"]):
    if download:
        print("Starting Download, this might take a while..")
        _download(dataset_select, out_dir=out_path)
        print("Download finished! Data processing started.")

    for ds in dataset_select:
        (out_path/ds).mkdir(parents=True, exist_ok=True)
        df=_individual_prepare_data(out_path, ds)
        df.to_csv(out_path/ds/"series.csv", index=False)
        df=_individual_prepare_adj(out_path, ds)
        df.to_csv(out_path/ds/"edges.csv", index=False)

    if cleanup:
        _cleanup(out_path)
    
if __name__=="__main__":
    prepare()

