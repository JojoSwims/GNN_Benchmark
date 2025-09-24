from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import zipfile
import urllib.request
"""
This file concerns itself with the multidimensional PeMs datasets, those that have both a speed and a volume value.
"""

URL="https://zenodo.org/api/records/7816008/files-archive"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/pems_volume")
default_set=[]

def _download(zip_path = ZIP_DEFAULT_PATH, out_dir = OUT_DEFAULT_PATH):
    #Download:
    urllib.request.urlretrieve(URL, str(zip_path))

    #Unzip:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        print("Unzipped to", out_dir)
    
    #Remove unnecessary downloads
    try:
        shutil.rmtree(out_dir/"__MACOSX")
    except FileNotFoundError:
        pass

    #Remove the zip file after extraction
    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass


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
def _individual_prepare(out_dir, dataset):
    name="PEMS"+dataset
    if dataset=="04":
        date="2018-01-01 00:00:00"
    elif dataset=="08":
        date="2016-07-01 00:00:00"
    else: raise NotImplementedError("Only PEMS04 and 08 are implemented so far.")
    with np.load(out_dir/(name+".npz")) as data:
        #Shape of arr is (x,y,3)
        arr = data["data"]
        
    # Creates on indexed dataframe for each of the three dimensions
    idx = pd.date_range(date, periods=arr.shape[0], freq="5min")
    dfs = [pd.DataFrame(x.squeeze(-1), index=idx) for x in np.split(arr, 3, axis=-1)]
    
    #Transform those dataframes into our long format.
    names = ['flow','occupancy','speed']
    longs = [df.stack().rename(names[i]).rename_axis(['ts','node']).reset_index().set_index('ts')[ [names[i],'node'] ] for i, df in enumerate(dfs)]
    longs = [d.reindex(columns=['node', names[i]]) for i, d in enumerate(longs)]
    
    #Combine them into one dataframe:
    df= pd.concat([d.set_index('node', append=True) for d in longs], axis=1).rename_axis(['ts','node']).sort_index()
    
    #Output it to our directory
    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True) 
    df.to_csv(dest / "series.csv")
    
    #Delete empty lines from outdire/name+".csv" sure distancescsv has right form (delete empty lines)
    src = out_dir / f"{name}.csv"
    cleaned = dest / "edges.csv"
    cleaned.write_text("\n".join(ln for ln in src.read_text().splitlines() if ln.strip()))    

def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH, dataset_select: list =["04", "08"]):
    if download:
        print("Starting Download, this might take a while..")
        _download(out_dir=out_path)

    for ds in dataset_select:
        _individual_prepare(out_path, ds)

    if cleanup:
        _cleanup(out_path)

if __name__=="__main__":
    prepare(download=True)