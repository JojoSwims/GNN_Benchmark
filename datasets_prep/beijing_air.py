from pathlib import Path
import zipfile
import urllib.request
import shutil
import pandas as pd
from math import radians, sin, cos, asin, sqrt

"""
#TODO: Figure out how to remove the intermediate Data folder when working on the downloaded Data
Some info on the raw data downloaded and how it is parsed:
There are 437 measuring stations but these are grouped in two clusters:
Beijing-Tianjin (Cluster 1) and Shenzhen-Guangzhou (Cluster 2)
-city.csv is a list of all city names, their city id and their respective Cluster.
-district.csv links districts to city ids (multiple districts per city).
-Finally, stations.csv links node ids to districts 
(which, through the above, also links them to a city and a cluster.)
=>This implies there are multiple ways to subdivide the dataset=> By city or by Cluster.
-stations.csv also has the associated latitude and longitude to allow distance calculations.
Finally for node features, 6 features and 2 exogenous variable categories are provided
Features:
Measuremnts of particle concentration for PM25, PM10, NO2, CO, O3, SO2
=>This allows the calculation of an Air Quality index, which is one of the possible forecasting targets.
Exogenous Variables:
-Actual Weather:  
Weather (encoded as numerical, e.g. 1 for Sunny, 2 for Cloudy), Temperature, Pressure, Humidity, Wind Speed, Wind Direction
-Forecasted Weather, the weather forecast at time t for time t+h:
Weather, Up Temperature, Bottom Temperature, Wind Level, Wind direction
---
This is a very rich dataset that allows testing how models react to the addition/removal of features and exogenous variables.
For now, we implement a minimal version of the dataset =>Only the PM2.5 concentration, for the whole dataset, the city of Beijin, Cluster 1 or Cluster 2.
"""

URL="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Data-1.zip"
ZIP_DEFAULT_PATH = Path("./../temp/data.zip")
OUT_DEFAULT_PATH = Path("./../temp/aqi")
#List of features that we keep from the raw data:
KEPT_FEATURES=["PM25_Concentration"]

def _download(zip_path = ZIP_DEFAULT_PATH, out_dir = OUT_DEFAULT_PATH):
    #Download:
    urllib.request.urlretrieve(URL, str(zip_path))

    #Unzip:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    # Flatten: look for a single top folder containing a "Data" dir, move it up, remove the wrapper 
    for top in (p for p in out_dir.iterdir() if p.is_dir()): 
        inner = top / "Data" 
        if inner.is_dir(): 
            dst = out_dir / "Data" 
            if dst.exists(): 
                shutil.rmtree(dst) 
                shutil.move(str(inner), str(out_dir)) 
                shutil.rmtree(top) 
                break # done
    #Move folders out of data:
    data_dir=out_dir/"Data"
    for item in data_dir.iterdir():
        shutil.move(str(item), str(out_dir))
    shutil.rmtree(data_dir)
    #Remove unnecessary downloads:
    try:
        (out_dir/"readme.pdf").unlink()
    except FileNotFoundError:
        pass

    #Remove the zip file after extraction
    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass

    print(f"Unzipped to: {out_dir.resolve()}")

#TODO: Could consider returning the used nodes, districts and cities, this would make integration with exogenous variables easier.
def _get_node_list(working_dir, subdivision):
    stations=pd.read_csv(working_dir/"station.csv")
    
    if subdivision=="beijing":
        # Districts that correspond to Beijing
        district_ids = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116]
        ids = stations.loc[stations["district_id"].isin(district_ids), "station_id"].tolist()
    elif subdivision=="cluster1" or subdivision=="cluster2":
        cluster_number=int(subdivision[-1])
        
        #Select the city ids that are in our cluster
        cities=pd.read_csv(working_dir/"city.csv")
        city_ids= cities.loc[cities["cluster_id"] == cluster_number, "city_id"].tolist()
        
        #Select the distric ids that are in the cities that are in our cluster
        districts=pd.read_csv(working_dir/"district.csv")
        district_ids=districts.loc[districts["city_id"].isin(city_ids), "district_id"].tolist()
        
        #Select the stations that are in our district
        ids = stations.loc[stations["district_id"].isin(district_ids), "station_id"].tolist()

    elif subdivision=="all":
        ids = stations["station_id"].tolist()
    else:
        raise NotImplementedError(subdivision, "is not a valid subdivision, please enter one of beijing, cluster1, cluster2, or all.")
    return ids

    #Converts our airquality data to our intermediate representation, only keeping the node_ids and features we need.


def _get_distance_from_coords(lat1, long1, lat2, long2):
    """
    Great-circle distance between two coordinates using the haversine formula.
    Parameters
    ----------
    lat1, long1 : float
        Latitude and longitude of the first point in decimal degrees.
    lat2, long2 : float
        Latitude and longitude of the second point in decimal degrees.
    Returns
    -------
    float
        Distance in meters.
    """
    # Mean Earth radius (WGS84): meters
    R = 6371008.8

    # Convert degrees to radians
    φ1, λ1, φ2, λ2 = map(radians, (lat1, long1, lat2, long2))

    # Haversine formula
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ / 2)**2 + cos(φ1) * cos(φ2) * sin(dλ / 2)**2
    c = 2 * asin(sqrt(a))

    return R * c

def _get_adj_matrix(out_path, nodes):
    df = pd.read_csv(out_path/"station.csv",dtype={"station_id": "int64", "latitude": "float64", "longitude": "float64"})
    df = df[df["station_id"].isin(nodes)]
    df = df[["station_id", "latitude", "longitude"]].reset_index(drop=True)
    
    rows = []
    # Nested loop; add symmetric rows; skip self-pairs
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if row_i["station_id"] == row_j["station_id"]:
                continue
            d = _get_distance_from_coords(row_i["latitude"], row_i["longitude"],
                                          row_j["latitude"], row_j["longitude"])
            rows.append((row_i["station_id"], row_j["station_id"], d))
    edges= pd.DataFrame(rows, columns=["from", "to", "cost"])
    edges = edges.astype({"from": "int64", "to": "int64", "cost": "float64"})
    return edges

def _convert_to_ir(in_path, out_path, node_ids, features=KEPT_FEATURES):
    df = pd.read_csv(in_path, parse_dates=[1])
    node_col, ts_col = df.columns[0], df.columns[1]

    df = df[df[node_col].isin(node_ids)]

    #Remove a few values with errors in dates:
    df = df[df[ts_col] > pd.Timestamp("2014-01-01")] 

    feat_cols = [c for c in features if c in df.columns[2:]]
    df = df[[ts_col, node_col, *feat_cols]].copy()
    df.rename(columns={ts_col: "ts", node_col: "node_id"}, inplace=True)
    ts_col, node_col = "ts", "node_id"

    # Treat zeros as missing values as soon as we have the feature columns
    if feat_cols:
        df.loc[:, feat_cols] = df.loc[:, feat_cols].where(df.loc[:, feat_cols] != 0)

    # Sort by timestamp, then node_id
    df = df.sort_values(by=[ts_col, node_col]).reset_index(drop=True)

    df.to_csv(out_path/"series.csv", index=False)

    # Get the mask (True indicates an observed/original value) and output it:
    observation_mask = df[[ts_col, node_col]].join(df[feat_cols].notna())
    observation_mask.to_csv(out_path/"mask.csv", index=False)

    # Get the adjacency matrix
    adjacency_matrix=_get_adj_matrix(out_path, node_ids)
    adjacency_matrix.to_csv(out_path / "edges.csv", index=False)
    
def _cleanup(working_dir):
    keep = {"series.csv", "mask.csv", "edges.csv"}
    for p in working_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv" and p.name not in keep:
            p.unlink()



def prepare(download=True, cleanup=True, out_path=None, subdivision="all"):
    #Subdivisions: "all", "cluster1", "cluster2", "beijing"
    if out_path!=None:
        out_dir=out_path
    else:
        out_dir=OUT_DEFAULT_PATH
    if download:
        print("Downloading data...")
        _download(out_dir=out_dir)

    node_ids=_get_node_list(out_dir, subdivision)
    _convert_to_ir(in_path=out_dir/"airquality.csv", out_path=out_dir, node_ids=node_ids)
    if cleanup:
        _cleanup(out_dir)

if __name__=="__main__":
    prepare(subdivision="cluster1")