from pathlib import Path
import urllib.request
import zipfile
import shutil
import pandas as pd



"""
Dataset description:
See https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (also includes the paper to cite.)
"""

URL="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/elergone")
DOWNLOAD_NAME="LD2011_2014.txt"

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



def prepare(download=True, cleanup=True, out_path=OUT_DEFAULT_PATH):
    if download:
        print("Downloading data...")
        _download(out_dir=out_path)
    print("Preparing the data, this will take a while.")
    df=pd.read_csv(
        out_path / DOWNLOAD_NAME,
        sep=";",                 # semicolon delimiter
        quotechar='"',           # quoted fields
        decimal=",",             # comma decimals
        parse_dates=[0],         # first column is datetime
        index_col=0,             # make it the index (so math is clean)
    )
    df.index.name = "timestamp"

    #Dividing by 4 makes the value go from kW to kWh (described in the dataset description above)
    df=df/4.0

    # Replace zero measurements with NaN before reshaping the dataset
    if not df.empty:
        value_columns = df.columns.tolist()
        df.loc[:, value_columns] = df.loc[:, value_columns].where(df.loc[:, value_columns] != 0)

    #Pivot the dataframe to our intermediate representation:
    df = (
        df.stack()
          .rename_axis(["ts", "node_id"])
          .reset_index(name="value")
          .sort_values("ts", kind="mergesort")
          .reset_index(drop=True)
    )

    #Output our intermediate representation:
    df.to_csv(out_path/"series.csv", index=False)

    #Output our mask:
    id_cols   = df.columns[:2]
    feat_cols = df.columns[2]
    observation_mask = df[id_cols].join(df[feat_cols].notna())
    observation_mask.to_csv(out_path/"mask.csv", index=False)
    
    if cleanup:
        try:
            (out_path / DOWNLOAD_NAME).unlink()
        except FileNotFoundError:
            pass

if __name__=="__main__":
    prepare(download=True)