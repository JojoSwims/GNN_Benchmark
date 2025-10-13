from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import zipfile
import urllib.request
"""
This file concerns itself with the multidimensional PeMs datasets, those that have both a speed and a volume value.
"""

PEMS08_NODE_ORDER=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]
PEMS04_NODE_ORDER=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306]


URL="https://zenodo.org/api/records/7816008/files-archive"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/pems_volume")
DEFAULT_SET=["04", "08"]

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
    
    N = arr.shape[1]
    # Creates on indexed dataframe for each of the three dimensions
    idx = pd.date_range(date, periods=arr.shape[0], freq="5min")
    dfs = [pd.DataFrame(x.squeeze(-1), index=idx) for x in np.split(arr, 3, axis=-1)]
    dfs = [df.where(df != 0) for df in dfs]
    
    #Transform those dataframes into our long format.
    names = ['flow','occupancy','speed']
    longs = [df.stack().rename(names[i]).rename_axis(['ts','node']).reset_index().set_index('ts')[ [names[i],'node'] ] for i, df in enumerate(dfs)]
    longs = [d.reindex(columns=['node', names[i]]) for i, d in enumerate(longs)]
    
        # Combine channels (same), but we will reindex to full grid right after (NEW)
    df = (
        pd.concat([d.set_index("node", append=True) for d in longs], axis=1)
          .rename_axis(["ts", "node"])
          .sort_index()
    )


    full_grid = pd.MultiIndex.from_product([idx, range(N)], names=["ts", "node"])
    df = df.reindex(full_grid)


    df = df.reset_index().rename(columns={"node": "node_id"}).sort_values(["ts", "node_id"])


    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "series.csv", index=False)

    mask = df[["ts", "node_id"]].copy()
    for c in names:
        mask[c] = df[c].notna()
    mask.to_csv(dest / "mask.csv", index=False)

    src = out_dir / f"{name}.csv"
    cleaned = dest / "edges.csv"
    cleaned.write_text("\n".join(ln for ln in src.read_text().splitlines() if ln.strip()))

def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH, dataset_select: list =DEFAULT_SET):
    if download:
        print("Starting Download, this might take a while..")
        _download(out_dir=out_path)

    for ds in dataset_select:
        _individual_prepare(out_path, ds)

    if cleanup:
        _cleanup(out_path)

if __name__=="__main__":
    prepare(download=True)