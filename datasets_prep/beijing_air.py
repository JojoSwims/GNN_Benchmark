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
BEIJING_NODE_ORDER=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036]
CLUSTER1_NODE_ORDER=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6010, 6011, 6012, 6013, 6014, 6015, 6016, 6017, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014, 11015, 11016, 11017, 11018, 11019, 11020, 11021, 11022, 11023, 11024, 11025, 12001, 13001, 13003, 13004, 13005, 13006, 13007, 13008, 13009, 13010, 13011, 13012, 13013, 13014, 14001, 14002, 14003, 14004, 14005, 14006, 14007, 14008, 17001, 17002, 17003, 17004, 17005, 17007, 17008, 17009, 17010, 17011, 17012, 18001, 19001, 19002, 19003, 19004, 19005, 19006, 19007, 19008, 19009, 19010, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 21001, 21002, 21003, 21004, 21005, 21006, 21007, 22001, 22002, 22003, 22004, 22005, 22006, 22007, 22008, 22009, 22010, 23001, 23002, 23003, 23004, 23005, 6028, 13002, 17006, 128001, 128002, 128003, 128004, 128005, 128006, 128007, 149001, 149004, 149006, 149007, 149009, 149010, 149014, 149015, 151003, 151004, 151007, 151009, 151011, 152001, 152004, 159001, 159002, 159003, 160001, 160005, 160006, 152002, 159004, 160003, 160004, 160007, 160008, 160009, 159005, 151002, 151005, 151006, 151008, 151010, 151012, 151013, 152003, 152005, 152006, 159006, 13015, 13016, 13017, 13018, 14009, 17013, 17014, 17015, 17016, 17017, 17018, 17019, 17020, 17021, 17022, 17023, 17024, 17025, 17026, 17027, 17028, 19011, 19012, 19013, 19014, 19015, 19016, 19017, 19018, 19019, 19020, 20010, 20011, 20012, 20013, 20014, 21008, 21009, 21010, 21011, 21012, 21013, 21014, 21015, 21016, 22011, 22012, 23006, 23007, 23008, 23009, 23010, 23011, 23012, 23013, 149002, 149003, 149005, 149008, 149011, 149012, 149013, 6040]
CLUSTER2_NODE_ORDER=[4003, 4007, 4008, 4009, 4011, 4014, 4017, 4018, 4019, 4020, 9017, 9018, 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027, 9028, 9029, 9030, 9031, 9032, 9033, 9034, 9035, 9036, 9037, 9038, 9039, 9040, 9041, 9042, 9043, 9044, 9045, 9046, 10001, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10015, 10016, 24001, 24002, 24003, 25001, 25002, 25003, 25004, 26001, 26002, 26003, 26004, 26005, 26006, 26007, 26008, 27001, 27002, 27003, 28001, 28002, 28003, 28004, 28005, 28006, 28007, 29001, 29003, 29004, 29005, 29006, 29007, 30002, 30003, 30004, 31001, 31002, 31003, 31004, 32001, 32002, 32003, 33001, 33002, 33003, 34001, 34002, 34003, 34004, 34005, 34006, 35001, 35002, 35003, 36001, 36002, 36003, 36004, 36005, 37001, 37002, 37003, 38001, 38002, 38003, 40001, 40002, 40003, 41001, 41002, 41003, 41004, 42001, 42002, 42003, 42004, 9058, 9047, 4002, 9016, 9059, 9060, 9061, 9062, 371001, 371002, 371003, 371004, 372001, 372002, 311001, 311002, 311003, 311004, 311005, 9064, 9065, 9066, 9067, 25006, 27004, 27005, 27006, 30005, 30006, 30007, 33004, 9063, 29008]

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
    prepare(subdivision="cluster2")