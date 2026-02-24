import urllib.request
import zipfile
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from math import radians, sin, cos, asin, sqrt

URL = "https://github.com/uctb/Urban-Dataset/raw/main/Public_Datasets/Pedestrian/60_minutes/Pedestrian_Melbourne.zip"

NODE_ORDER = [1,2,3,5,6,8,9,10,11,12,14,17,19,20,21,23,24,25,26,27,28,29,30,31,36,37,39,40,41,42,44,46,47,48,49,50,51,53,54,56,57,58,59,61,62,63,65,66,67,68,69,70,71,72,75]

START = "2021-01-01 00:00:00"
END   = "2022-11-01 00:00:00"
FREQ  = "60min"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1, lam1, phi2, lam2 = map(radians, (lat1, lon1, lat2, lon2))
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * R * asin(sqrt(a))


if __name__ == "__main__":
    tmpdir = Path("tmp")
    tmpdir.mkdir(exist_ok=True)

    zip_path = tmpdir / "data.zip"
    extract_dir = tmpdir / "extracted"

    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    pkl_file = next(extract_dir.rglob("*.pkl"))
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    node = data["Node"]
    X_all = node["TrafficNode"]       # [T, N_total]
    info = node["StationInfo"]        # N_total entries: [id, date, lat, lon, name]

    # Quick alignment check
    T_all, N_all = X_all.shape
    if N_all != len(info):
        raise ValueError(f"Alignment fail: TrafficNode N={N_all} != len(StationInfo)={len(info)}")

    # Build stations table (preserves StationInfo order via index)
    stations_all = pd.DataFrame(info, columns=["node_id", "date", "lat", "lon", "name"])
    stations_all["node_id"] = stations_all["node_id"].astype(int)

    # Map station_id -> column index (StationInfo order)
    id2col = {sid: i for i, sid in enumerate(stations_all["node_id"].tolist())}

    # Enforce deterministic order, keep only present nodes
    node_ids = [sid for sid in NODE_ORDER if sid in id2col]
    cols = [id2col[sid] for sid in node_ids]

    # Subset data + stations to chosen nodes, in NODE_ORDER
    X = X_all[:, cols]
    stations = stations_all.set_index("node_id").loc[node_ids].reset_index()

    # Save node order actually used
    pd.Series([str(i) for i in node_ids], name="node_id").to_csv("node_order.csv", index=False)

    # ---- distances.csv ----
    rows = []
    for i in range(len(stations)):
        for j in range(len(stations)):
            if i == j:
                continue
            d = haversine_m(
                float(stations.at[i, "lat"]),
                float(stations.at[i, "lon"]),
                float(stations.at[j, "lat"]),
                float(stations.at[j, "lon"]),
            )
            rows.append((str(int(stations.at[i, "node_id"])), str(int(stations.at[j, "node_id"])), d))
    pd.DataFrame(rows, columns=["src", "dst", "cost"]).to_csv("distances.csv", index=False)

    # ---- series.csv ----
    ts = pd.date_range(START, END, freq=FREQ, inclusive="left")
    if len(ts) != X.shape[0]:
        ts2 = pd.date_range(START, END, freq=FREQ, inclusive="both")
        raise ValueError(
            f"Timestamp count mismatch: TrafficNode T={X.shape[0]}, ts(left)={len(ts)}, ts(both)={len(ts2)}"
        )

    T, N = X.shape
    series = pd.DataFrame({
        "ts": np.repeat(ts.to_numpy(), N),
        "node_id": np.tile(np.array([str(i) for i in node_ids], dtype=object), T),
        "count": X.reshape(-1),
    })
    series.to_csv("series.csv", index=False)

    shutil.rmtree(tmpdir)